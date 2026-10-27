"""Stage-1 2DoF training (and eval-only) entry point for VIGOR.

Usage examples are in ``vigor_main/README.md``. Typical training:

    python -m vigor_main.train --area same --GPS_error_coe 1 --lr 1.25e-4 --name reproduce

Typical eval-only:

    python -m vigor_main.train --test 1 --area same --ckpt /path/to/model_X.pth \\
        --GPS_error_coe 1 --lr 1.25e-4 --name verify
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import os
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR

# Avoid "Too many open files" — VIGOR's test sets are 50k+ samples and the
# default file_descriptor sharing strategy leaks fds across dataloader workers.
torch.multiprocessing.set_sharing_strategy("file_system")

from . import config
from .data import load_test, load_train_and_val
from .eval import evaluate
from .losses import Weakly_supervised_loss_w_GPS_error, batch_wise_cross_corr
from .model import BevSplatVIGOR


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="BevSplat VIGOR 2DoF trainer")

    p.add_argument("--area", type=str, choices=["same", "cross"], default="same",
                   help="VIGOR test protocol — same-area or cross-area.")

    p.add_argument("--rotation_range", type=float, default=config.ROTATION_RANGE,
                   help="0 for the standard 2DoF protocol; non-zero adds rotation perturbation.")

    p.add_argument("--GPS_error_coe", type=float, default=0.0,
                   help="λ₁ in the paper. 0=Weakly-only, 1=add the GPS-error term.")
    p.add_argument("--GPS_error", type=float, default=config.GPS_ERROR_RADIUS_M)

    p.add_argument("--lr", type=float, default=config.LR,
                   help="Default 1.25e-4 (Same-Area). Use 1e-4 or 6.5e-5 for Cross-Area variants.")
    p.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    p.add_argument("--epochs", type=int, default=config.EPOCHS)
    p.add_argument("--amount", type=float, default=config.DATA_AMOUNT,
                   help="Fraction of training data to use (legacy --amount).")

    p.add_argument("--resume", type=int, default=0,
                   help="Epoch index to resume from (loads model_{resume-1}.pth from save_path).")

    p.add_argument("--continue_weights", type=str, default="",
                   help="Strict weights continuation; AdamW/RNG reset, original schedule clock restored.")
    p.add_argument("--completed_steps", type=int, default=0)

    p.add_argument("--name", type=str, default="reproduce",
                   help="Tag appended to the auto-generated save directory.")

    p.add_argument("--test", type=int, default=0,
                   help="If 1, load --ckpt and evaluate on the test split only.")
    p.add_argument("--ckpt", type=str, default="",
                   help="Checkpoint .pth to load (required when --test 1).")

    p.add_argument("--max_batches", type=int, default=0,
                   help="If > 0, stop training after this many batches per epoch (smoke test).")

    p.add_argument("--cuda", type=int, default=0,
                   help="CUDA device index (replaces the legacy hardcoded CUDA_VISIBLE_DEVICES=3).")

    p.add_argument("--init_ckpt", type=str, default="",
                   help="Strict full-model initialization for a fresh optimizer/schedule.")
    p.add_argument("--freeze_gaussian_bn", type=int, choices=[0, 1], default=0,
                   help="Fix Gaussian BN statistics only; all affine/encoder parameters still train.")
    p.add_argument("--gaussian_lr_scale", type=float, default=1.0)
    p.add_argument("--geometry_prior_coe", type=float, default=0.0)
    p.add_argument("--zero_init_gaussian_offsets", type=int, choices=[0, 1], default=0)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--pilot_steps", type=int, default=0,
                   help="Stop after exactly this many optimizer steps, then full internal val.")
    p.add_argument("--eval_split", choices=["test", "val"], default="test")
    p.add_argument("--model_behavior", choices=["legacy", "corrected"], default="legacy",
                   help="Opt-in opacity routing and single-sigmoid confidence; legacy for old weights.")
    p.add_argument("--translation_origin", choices=["legacy", "crop"], default="legacy",
                   help="Correlation decoder origin; crop matches actual center_crop integer offset.")
    p.add_argument("--confidence_mode", choices=["inherit", "legacy", "corrected"], default="inherit")
    p.add_argument("--gaussian_pretrained_lr_scale", type=float, default=None,
                   help="Optional multiplier of Gaussian LR for public-pretrained ResNet only.")
    p.add_argument("--gaussian_offset_max", type=float, default=3.0)
    p.add_argument("--gaussian_scale_max", type=float, default=0.3)
    p.add_argument("--gaussian_opacity_max", type=float, default=1.0,
                   help="Smooth opacity range: maximum times sigmoid, not clipping.")
    args = p.parse_args(argv)
    for key in ("gaussian_pretrained_lr_scale", "gaussian_offset_max", "gaussian_scale_max", "gaussian_opacity_max"):
        value = getattr(args, key)
        if value is not None and (not math.isfinite(value) or value <= 0):
            p.error("--" + key + " must be finite and positive")
    if args.gaussian_opacity_max > 1:
        p.error("--gaussian_opacity_max must not exceed one")
    if not math.isfinite(args.gaussian_lr_scale) or args.gaussian_lr_scale <= 0:
        p.error("--gaussian_lr_scale must be finite and positive")
    if not math.isfinite(args.geometry_prior_coe) or args.geometry_prior_coe < 0:
        p.error("--geometry_prior_coe must be finite and nonnegative")
    if args.pilot_steps < 0 or (args.pilot_steps and (args.max_batches or args.resume or args.test)):
        p.error("--pilot_steps is nonnegative and incompatible with max_batches/resume/test")
    if args.zero_init_gaussian_offsets and (args.resume or args.test):
        p.error("Position zero initialization is for fresh training only")

    if args.init_ckpt and (args.resume or args.test or args.zero_init_gaussian_offsets):
        p.error("--init_ckpt is fresh-only and incompatible with resume/test/position zero-init")

    if args.continue_weights:
        if (args.test or args.init_ckpt or args.pilot_steps or args.max_batches or
                args.zero_init_gaussian_offsets or args.resume <= 0 or
                args.resume >= args.epochs or args.completed_steps <= 0 or args.seed is None):
            p.error("--continue_weights requires 0<resume<epochs, completed_steps, seed; no test/init/pilot/max_batches/zero-init")
        if not Path(args.continue_weights).is_file():
            p.error("--continue_weights source does not exist")
    elif args.completed_steps:
        p.error("--completed_steps requires --continue_weights")

    # Inject the legacy single-value flags so the algorithm modules
    # receive the exact Namespace they expect.
    args.stage = 3                                  # legacy fixed value used by dataloader
    args.level = config.LEVEL_STR
    args.channels = config.CHANNELS_STR
    args.share = config.SHARE_FEATURE_NET
    args.ConfGrd = config.CONF_GRD
    args.ConfSat = config.CONF_SAT
    args.N_iters = config.N_ITERS
    args.proj = config.PROJ
    args.Optimizer = config.OPTIMIZER_KIND
    args.task = "2DoF"
    args.use_uncertainty = config.USE_UNCERTAINTY
    args.Supervision = config.SUPERVISION
    args.grd_res = config.GRD_RES
    args.visualize = 0
    args.multi_gpu = 0
    args.shift_range_lat = 0.0
    args.shift_range_lon = 0.0
    return args


# ---------------------------------------------------------------------------
# Train / Eval
# ---------------------------------------------------------------------------

def _digest_tensors(items, omit_position=False):
    digest = hashlib.sha256()
    for name, tensor in items:
        value = tensor.detach().cpu().contiguous()
        if omit_position and name in ("gaussian_encoder.to_gaussians.1.weight",
                                      "gaussian_encoder.to_gaussians.1.bias"):
            value = value.clone()
            for start in (0, 11, 22):
                value[start:start + 3].zero_()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


class _TraceSampler:
    """Observe the actual sampler including prefetch; never draw a second order."""
    def __init__(self, sampler, subset):
        self.sampler, self.subset, self.prefix = sampler, subset, []

    def __len__(self):
        return len(self.sampler)

    def __iter__(self):
        for index in self.sampler:
            if len(self.prefix) < 8000:
                self.prefix.append(int(self.subset.indices[index]))
            yield index


def _set_training_mode(net, args):
    net.train()
    if args.freeze_gaussian_bn:
        for module in net.gaussian_encoder.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.eval()


def _gaussian_bn_audit(encoder):
    modules = [(name, module) for name, module in encoder.named_modules()
               if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)]
    return {"buffers_sha256": _digest_tensors(
                (name + "." + key, value) for name, module in modules
                for key, value in module.named_buffers(recurse=False)),
            "counters": {name: int(module.num_batches_tracked.item()) for name, module in modules
                         if module.num_batches_tracked is not None}}


def _record_validation(history, metrics, epoch, loader, save_path):
    if not isinstance(metrics, dict):
        return
    if not all(math.isfinite(float(value)) for value in metrics.values()):
        raise RuntimeError("Nonfinite full-validation metric")
    history.append(dict(epoch=epoch, samples=len(loader.dataset), metrics=metrics))
    (save_path / "validation_history.json").write_text(json.dumps(history, indent=2) + "\n")


def _audit_initial(net, loader, save_path):
    subset = loader.dataset
    dataset = subset.dataset
    identities = [(int(i), dataset.grd_list[i], int(dataset.label[i][0]),
                   dataset.delta[i][0].tolist()) for i in subset.indices]
    result = {"initial_full_state": _digest_tensors(net.state_dict().items()),
              "initial_gaussian_bn": _gaussian_bn_audit(net.gaussian_encoder),
              "initial_nonposition_state": _digest_tensors(net.state_dict().items(), True),
              "train_members": hashlib.sha256(json.dumps(identities).encode()).hexdigest(),
              "train_samples": len(subset),
              "initial_parameters": {name: _digest_tensors(module.named_parameters()) for name, module in
                                     (("gaussian", net.gaussian_encoder), ("dpt_sat", net.dpt_sat),
                                      ("dpt_grd", net.dpt_grd), ("dino", net.dino_feat))}}
    (save_path / "initial_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def _audit_finish(net, audit, trace, steps, scheduler, save_path):
    if audit is None:
        return
    audit = dict(audit, final_gaussian_bn=_gaussian_bn_audit(net.gaussian_encoder), optimizer_steps=steps, scheduler_total_steps=scheduler.total_steps,
                 scheduler_steps=scheduler.last_epoch,
                 consumed_prefix_count=min(steps * 8, len(trace.prefix)),
                 consumed_prefix_sha256=hashlib.sha256(json.dumps(trace.prefix[:min(steps * 8, 8000)]).encode()).hexdigest(),
                 final_parameters={name: _digest_tensors(module.named_parameters()) for name, module in
                                   (("gaussian", net.gaussian_encoder), ("dpt_sat", net.dpt_sat),
                                    ("dpt_grd", net.dpt_grd), ("dino", net.dino_feat))})
    if "inherited_steps" in audit:
        audit["new_optimizer_steps"] = steps - audit["inherited_steps"]
    audit["finite_state"] = all(torch.isfinite(t).all().item() for t in net.state_dict().values())
    if not audit["finite_state"]:
        raise RuntimeError("Nonfinite training state")
    (save_path / "training_audit.json").write_text(json.dumps(audit, indent=2) + "\n")


def _restore_continuation_clock(net, optimizer, scheduler, completed_steps):
    """Restore only LR time. Missing Adam moments and RNG are explicitly reset."""
    if optimizer.state or not 0 < completed_steps < scheduler.total_steps:
        raise RuntimeError("Continuation requires empty AdamW and a valid inherited clock")
    before = _digest_tensors(net.state_dict().items())
    if scheduler.last_epoch != 0:
        raise RuntimeError("Continuation scheduler must be newly constructed at zero")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r"Detected call of `lr_scheduler.step\(\)` before `optimizer.step\(\)`")
        for _ in range(completed_steps):
            scheduler.step()
    if optimizer.state or scheduler.last_epoch != completed_steps or before != _digest_tensors(net.state_dict().items()):
        raise RuntimeError("Scheduler-only restoration changed model/Adam or clock")
    return {"inherited_steps": completed_steps, "initial_full_state": before,
            "scheduler_steps": scheduler.last_epoch, "scheduler_total_steps": scheduler.total_steps,
            "initial_lrs": scheduler.get_last_lr(), "adam_moments_reset": True,
            "rng_reseeded_not_restored": True}


def _save_training_state(net, optimizer, scheduler, args, steps, next_epoch, save_path):
    """Atomic epoch boundary snapshot; only tensors and weights_only-safe builtins."""
    numpy_state = np.random.get_state()
    state = {"format": "vigor-training-state-v1", "model": net.state_dict(),
             "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
             "global_step": steps, "next_epoch": next_epoch, "effective_args": vars(args),
             "python_rng": random.getstate(),
             "numpy_rng": {"kind": str(numpy_state[0]),
                           "keys": torch.tensor(numpy_state[1].astype(np.int64)),
                           "position": int(numpy_state[2]), "has_gauss": int(numpy_state[3]),
                           "cached_gaussian": float(numpy_state[4])},
             "torch_rng": torch.get_rng_state(),
             "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}
    target = save_path / "training_state_latest.pth"
    temporary = save_path / "training_state_latest.pth.tmp"
    torch.save(state, temporary)
    os.replace(temporary, target)


def _train(net, args, save_path: Path, device):
    save_path.mkdir(parents=True, exist_ok=True)
    train_loader, val_loader = load_train_and_val(
        args.batch_size,
        area=args.area,
        rotation_range=args.rotation_range,
        amount=args.amount,
    )

    if args.share:
        params = net.dpt_sat.parameters()
    else:
        params = list(net.dpt_grd.parameters()) + list(net.dpt_sat.parameters())
    gaussian_params = list(net.gaussian_encoder.parameters())
    params = gaussian_params + list(params)
    max_lrs = args.lr
    if args.gaussian_lr_scale != 1.0:
        dpt_params = list(net.dpt_sat.parameters())
        if not args.share:
            dpt_params += list(net.dpt_grd.parameters())
        params = [{"params": gaussian_params, "lr": args.lr * args.gaussian_lr_scale},
                  {"params": dpt_params, "lr": args.lr}]
        max_lrs = [args.lr * args.gaussian_lr_scale, args.lr]
    if getattr(args, "gaussian_pretrained_lr_scale", None) is not None:
        pretrained = list(net.gaussian_encoder.backbone.resnet_backbone.model.parameters())
        pretrained_ids = {id(p) for p in pretrained}
        other_gaussian = [p for p in gaussian_params if id(p) not in pretrained_ids]
        dpt_params = list(net.dpt_sat.parameters())
        if not args.share:
            dpt_params += list(net.dpt_grd.parameters())
        max_lrs = [args.lr * args.gaussian_lr_scale * args.gaussian_pretrained_lr_scale,
                   args.lr * args.gaussian_lr_scale, args.lr]
        params = [{"params": values, "lr": rate, "group_name": label}
                  for values, rate, label in zip(
                      (pretrained, other_gaussian, dpt_params), max_lrs,
                      ("public_resnet", "other_gaussian", "dpt"))]
        actual = [id(p) for group in params for p in group["params"]]
        expected = [id(p) for p in gaussian_params + dpt_params]
        if len(actual) != len(set(actual)) or set(actual) != set(expected):
            raise RuntimeError("Optimizer parameter groups overlap or omit parameters")
    optimizer = optim.AdamW(
        params,
        lr=args.lr,
        weight_decay=config.WEIGHT_DECAY,
        eps=config.ADAM_EPS,
    )

    scheduler = OneCycleLR(
        optimizer,
        max_lr=max_lrs,
        steps_per_epoch=len(train_loader),
        epochs=args.epochs,
        anneal_strategy=config.ONECYCLE_ANNEAL,
        pct_start=config.ONECYCLE_PCT_START,
        cycle_momentum=False,
    )

    continuation = None
    if getattr(args, "continue_weights", ""):
        if args.completed_steps != args.resume * len(train_loader):
            raise RuntimeError("Inherited steps must equal complete prior epochs")
        continuation = _restore_continuation_clock(net, optimizer, scheduler, args.completed_steps)
        continuation.update(source=args.continue_weights,
                            source_sha256=hashlib.sha256(Path(args.continue_weights).read_bytes()).hexdigest(),
                            resume_epoch=args.resume, target_epochs=args.epochs)
        (save_path / "continuation_audit.json").write_text(json.dumps(continuation, indent=2) + "\n")
        print("[continuation] " + json.dumps(continuation))

    audit, trace = None, None
    if args.seed is not None:
        audit = _audit_initial(net, train_loader, save_path)
        if continuation is not None:
            audit["inherited_steps"] = args.completed_steps
            (save_path / "initial_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
        trace = _TraceSampler(train_loader.batch_sampler.sampler, train_loader.dataset)
        train_loader.batch_sampler.sampler = trace
        (save_path / "effective_args.json").write_text(json.dumps(vars(args), indent=2) + "\n")
        parameter_names = {id(p): name for name, p in net.named_parameters()}
        rates = max_lrs if isinstance(max_lrs, list) else [max_lrs]
        runtime = {
            "confidence_mode": getattr(net, "confidence_mode", getattr(net, "model_behavior", "legacy")),
            "predicted_opacity": getattr(net, "model_behavior", "legacy") == "corrected",
            "offset_max": net.gaussian_encoder.offset_max,
            "scale_max": net.gaussian_encoder.scale_max,
            "opacity_max": net.gaussian_encoder.opacity_max,
            "geometry_prior_divisor_m": 3.0,
            "optimizer_groups": [{"name": group.get("group_name", "gaussian" if i == 0 else "dpt"),
                                  "max_lr": rates[i],
                                  "parameters": [parameter_names[id(p)] for p in group["params"]]}
                                 for i, group in enumerate(optimizer.param_groups)],
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
            "tf32_cudnn": torch.backends.cudnn.allow_tf32,
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"),
            "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "device": str(device), "workers": train_loader.num_workers,
        }
        (save_path / "runtime.json").write_text(json.dumps(runtime, indent=2) + "\n")
    steps = getattr(args, "completed_steps", 0)
    validation_history = []
    epoch_training_history = []
    print(f"[optimizer] max_lrs={max_lrs} total_steps={scheduler.total_steps}")
    t0 = time.time()
    for epoch in range(args.resume, args.epochs):
        _set_training_mode(net, args)
        n_batches = len(train_loader)
        epoch_start_steps, epoch_samples, last_batch = steps, 0, 0
        print(f"[epoch {epoch}] batch_size={args.batch_size}, num_batches={n_batches}")

        for loop_idx, batch in enumerate(train_loader):
            if args.max_batches and loop_idx >= args.max_batches:
                print(f"[smoke] stopping after {loop_idx} batches")
                break
            optimizer.zero_grad()
            (
                grd, sat, depth_imgs, grd_ori,
                gt_shift_u, gt_shift_v, gt_rot, meter_per_pixel,
            ) = (item.to(device) for item in batch)

            sat_feat_d, sat_conf_d, g2s_feat_d, g2s_conf_d, sat_uncer_d = net(
                sat, grd, depth_imgs, grd_ori, meter_per_pixel,
                gt_rot, gt_shift_u, gt_shift_v,
                loop=loop_idx, save_dir=str(save_path),
            )
            corr_maps = batch_wise_cross_corr(
                sat_feat_d, sat_conf_d, g2s_feat_d, g2s_conf_d, args, sat_uncer_d
            )
            l_weakly, l_gps = Weakly_supervised_loss_w_GPS_error(
                corr_maps, gt_shift_u, gt_shift_v, args, meter_per_pixel, args.GPS_error,
            )
            loss = l_weakly + l_gps * args.GPS_error_coe
            l_geometry = net.gaussian_encoder.geometry_prior() if args.geometry_prior_coe else loss.new_zeros(())
            loss = loss + args.geometry_prior_coe * l_geometry
            if not torch.isfinite(loss).item():
                raise RuntimeError("Nonfinite training loss")
            loss.backward()
            if args.seed is not None:
                finite = [torch.isfinite(p.grad).all() for group in optimizer.param_groups
                          for p in group['params'] if p.grad is not None]
                if not finite or not torch.stack(finite).all().item():
                    raise RuntimeError("Nonfinite optimizer gradient")
            optimizer.step()
            scheduler.step()
            steps += 1
            if continuation is not None:
                last_batch = int(grd.shape[0])
                epoch_samples += last_batch

            if loop_idx % 10 == 9:
                cur_lr = scheduler.get_last_lr()[0]
                dt = time.time() - t0
                print(
                    f"[epoch {epoch} loop {loop_idx}] "
                    f"L_Weakly={l_weakly.item():.4f}  L_GPS={l_gps.item():.4f}  "
                    f"L_geometry={l_geometry.item():.6f} step={steps} lr={cur_lr:.2e} "
                    f"gaussian_lr={scheduler.get_last_lr()[0]:.2e} dpt_lr={scheduler.get_last_lr()[-1]:.2e} dt={dt:.1f}s"
                )
                t0 = time.time()
            if args.pilot_steps and steps == args.pilot_steps:
                ckpt_path = save_path / f"pilot_step_{steps}.pth"
                torch.save(net.state_dict(), ckpt_path)
                print(f"[pilot] optimizer_steps={steps} saved {ckpt_path}")
                _audit_finish(net, audit, trace, steps, scheduler, save_path)
                evaluate(net, val_loader, args, save_path, epoch,
                         split_name=f"{args.area}_val", device=device)
                _set_training_mode(net, args)
                return

        ckpt_path = save_path / f"model_{epoch}.pth"
        torch.save(net.state_dict(), ckpt_path)
        print(f"[epoch {epoch}] saved {ckpt_path}")

        if not args.max_batches:
            metrics = evaluate(net, val_loader, args, save_path, epoch,
                               split_name=f"{args.area}_val", device=device)
            _set_training_mode(net, args)
            if args.seed is not None:
                _record_validation(validation_history, metrics, epoch, val_loader, save_path)
            if continuation is not None:
                epoch_training_history.append(dict(epoch=epoch, updates=steps - epoch_start_steps,
                                                   samples=epoch_samples, last_batch=last_batch,
                                                   global_step=steps))
                (save_path / "epoch_training_history.json").write_text(json.dumps(epoch_training_history, indent=2) + "\n")
                _save_training_state(net, optimizer, scheduler, args, steps, epoch + 1, save_path)

    if args.pilot_steps:
        raise RuntimeError("Training schedule ended before requested pilot_steps")
    _audit_finish(net, audit, trace, steps, scheduler, save_path)


def _eval_only(net, args, save_path: Path, device):
    save_path.mkdir(parents=True, exist_ok=True)
    if args.eval_split == "val":
        _, test_loader = load_train_and_val(
            args.batch_size, area=args.area, rotation_range=args.rotation_range, amount=args.amount)
    else:
        test_loader = load_test(
            args.batch_size, area=args.area, rotation_range=args.rotation_range, amount=args.amount)
    print(f"[eval-only] {args.area}-area")
    evaluate(net, test_loader, args, save_path, epoch=-1,
             split_name=f"{args.area}_val" if args.eval_split == "val" else args.area, device=device)


# ---------------------------------------------------------------------------
# Weight loading (shape-tolerant)
# ---------------------------------------------------------------------------

def _load_initial(net, ckpt_path: str, *, label="init"):
    """Load one complete finite state, without silently mixing initialization weights."""
    if not ckpt_path or not Path(ckpt_path).is_file():
        raise FileNotFoundError(f"[{label}] checkpoint not found: {ckpt_path}")
    pre = torch.load(ckpt_path, weights_only=True, map_location="cpu")
    current = net.state_dict()
    if not isinstance(pre, dict) or set(pre) != set(current):
        raise RuntimeError(f"[{label}] requires the complete exact model state")
    for key, value in pre.items():
        if not isinstance(value, torch.Tensor) or value.shape != current[key].shape or value.dtype != current[key].dtype:
            raise RuntimeError(f"[{label}] tensor shape/dtype mismatch: {key}")
        if not torch.isfinite(value).all():
            raise RuntimeError(f"[{label}] non-finite checkpoint tensor: {key}")
    net.load_state_dict(pre, strict=True)
    if not all(torch.equal(value.detach().cpu(), pre[key]) for key, value in net.state_dict().items()):
        raise RuntimeError(f"[{label}] tensor equality failed")
    print(f"[{label}] loaded {len(pre)}/{len(pre)} params from {ckpt_path} (0 shape-skip)")


def _load_filtered(net, ckpt_path: str, label: str):
    if not ckpt_path or not Path(ckpt_path).is_file():
        print(f"[{label}] WARN: ckpt path not found: {ckpt_path}")
        return
    pre = torch.load(ckpt_path, weights_only=True, map_location="cpu")
    cur = net.state_dict()
    kept, skipped = {}, 0
    for k, v in pre.items():
        if k in cur and v.shape == cur[k].shape:
            kept[k] = v
        elif k in cur:
            print(f"[{label}] skip {k}: pretrained {tuple(v.shape)} vs current {tuple(cur[k].shape)}")
            skipped += 1
    cur.update(kept)
    net.load_state_dict(cur)
    print(f"[{label}] loaded {len(kept)}/{len(pre)} params from {ckpt_path} ({skipped} shape-skip)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None):
    args = parse_args(argv)
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)

    np.random.seed(2022 if args.seed is None else args.seed)
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    save_path = config.save_dir(
        area=args.area,
        rotation_range=args.rotation_range,
        lr=args.lr,
        gps_error_coe=args.GPS_error_coe,
        gps_error=args.GPS_error,
        share=args.share,
        conf_grd=args.ConfGrd,
        conf_sat=args.ConfSat,
        name=args.name,
    )
    print(f"save_path: {save_path}")

    net = BevSplatVIGOR(args, device=device).to(device)

    if args.test:
        if not args.ckpt:
            raise SystemExit("--test 1 requires --ckpt <path-to-model.pth>")
        _load_initial(net, args.ckpt, label="eval")
        _eval_only(net, args, save_path, device)
        return

    if args.init_ckpt:
        _load_initial(net, args.init_ckpt)

    if args.continue_weights:
        _load_initial(net, args.continue_weights)
    elif args.resume:
        resume_path = save_path / f"model_{args.resume - 1}.pth"
        _load_filtered(net, str(resume_path), label="resume")

    if args.zero_init_gaussian_offsets:
        net.gaussian_encoder.zero_initialize_position_offsets()
    _train(net, args, save_path, device)


if __name__ == "__main__":
    main()
