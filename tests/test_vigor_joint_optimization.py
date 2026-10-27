"""Focused optimizer/scheduler checks; opt-in full-size CUDA gradient smoke."""
import ast
import argparse
import runpy
import hashlib
import json
import math
import random
import numpy as np
import os
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn, optim
from torch.optim.lr_scheduler import OneCycleLR

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def trainer_ast(relative, name):
    tree = ast.parse((ROOT / relative).read_text(), filename=relative)
    return next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)


def optimizer_scope(relative, name, net, args, save_path, config=None):
    """Execute the actual trainer prefix, not a copied parameter expression."""
    fn = trainer_ast(relative, name)
    cutoff = 'time_start' if name == 'train' else 't0'
    body = []
    for node in fn.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == cutoff for t in node.targets):
            break
        body.append(node)
    scope = dict(net=net, args=args, save_path=save_path, optim=optim,
                 OneCycleLR=OneCycleLR, config=config, Path=Path,
                 load_train_and_val=lambda *a, **k: ([None] * 3, []))
    scope.update(training_helpers())
    exec(compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])), relative, 'exec'), scope)
    return scope


def training_helpers():
    scope = dict(torch=torch, hashlib=hashlib, json=json, math=math, random=random, np=np, Path=Path)
    tree = ast.parse((ROOT / 'vigor_main/train.py').read_text())
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))
             and n.name in ('_digest_tensors', '_TraceSampler', '_audit_initial', '_audit_finish',
                               '_set_training_mode', '_gaussian_bn_audit', '_record_validation', '_load_initial')]
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), 'vigor_main/train.py', 'exec'), scope)
    return scope


def args_for(share=0, supervision='Weakly'):
    return SimpleNamespace(share=share, Supervision=supervision, lr=1.25e-4,
                           batch_size=8, epochs=2, resume=0, area='same',
                           rotation_range=0.0, amount=1.0, max_batches=0,
                           GPS_error=5.0, GPS_error_coe=0.0, gaussian_lr_scale=1.0,
                           geometry_prior_coe=0.0, zero_init_gaussian_offsets=0, seed=None, pilot_steps=0,
                           init_ckpt="", freeze_gaussian_bn=0)


def small_config():
    return SimpleNamespace(BATCH_SIZE=8, WEIGHT_DECAY=5e-3, ADAM_EPS=1e-8,
                           ONECYCLE_STEPS_PER_EPOCH_BASE=2,
                           ONECYCLE_PCT_START=0.3, ONECYCLE_ANNEAL='cos')


class TinyNet(nn.Module):
    def __init__(self, share=0):
        super().__init__()
        self.share = share
        self.gaussian_encoder = nn.Linear(2, 2)
        self.dpt_sat = nn.Linear(2, 2)
        self.dpt_grd = nn.Linear(2, 2)
        self.dino_feat = nn.Linear(2, 2)

    def forward(self, sat, grd, *args, **kwargs):
        grd_head = self.dpt_sat if self.share else self.dpt_grd
        y = self.gaussian_encoder(sat) + self.dpt_sat(sat) + grd_head(grd)
        self.current_loss = y.square().mean()
        return ({}, {}, {}, {}, {})


class VigorOptimizerTests(unittest.TestCase):
    def test_strict_initialization_and_full_state_identity(self):
        load = training_helpers()['_load_initial']
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'init.pth'
            source, target = TinyNet(), TinyNet()
            torch.save(source.state_dict(), path)
            load(target, str(path))
            self.assertEqual(module_digest(source), module_digest(target))
            bad = dict(source.state_dict())
            bad.pop('gaussian_encoder.bias')
            torch.save(bad, path)
            with self.assertRaises(RuntimeError):
                load(target, str(path))
            bad = dict(source.state_dict())
            bad['gaussian_encoder.weight'] = bad['gaussian_encoder.weight'].double()
            torch.save(bad, path)
            with self.assertRaises(RuntimeError):
                load(target, str(path))

    def test_initialization_cli_conflicts_and_defaults(self):
        scope = dict(argparse=argparse, math=math, config=SimpleNamespace(**runpy.run_path(str(ROOT/'vigor_main/config.py'))))
        fn = trainer_ast('vigor_main/train.py', 'parse_args')
        exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])), 'vigor_main/train.py', 'exec'), scope)
        parse = scope['parse_args']
        defaults = parse([])
        self.assertEqual((defaults.init_ckpt, defaults.freeze_gaussian_bn), ("", 0))
        for flags in (['--test','1'], ['--resume','1'], ['--zero_init_gaussian_offsets','1']):
            with self.assertRaises(SystemExit):
                parse(['--init_ckpt','fixture.pth']+flags)

    def test_bn_statistics_policy_keeps_affine_trainable(self):
        mode = training_helpers()['_set_training_mode']
        for fixed in (0, 1):
            net = TinyNet()
            bn = nn.BatchNorm1d(2)
            net.gaussian_encoder = nn.Sequential(bn, nn.Linear(2, 2))
            args = args_for()
            args.freeze_gaussian_bn = fixed
            mode(net, args)
            before = bn.running_mean.clone()
            y = net.gaussian_encoder(torch.tensor([[1.,2.],[3.,7.],[4.,9.]]))
            y.square().sum().backward()
            self.assertTrue(net.gaussian_encoder.training)
            self.assertTrue(bn.weight.requires_grad and bn.bias.requires_grad)
            self.assertIsNotNone(bn.weight.grad)
            self.assertGreater(bn.weight.grad.abs().max().item(), 0)
            self.assertEqual(bn.num_batches_tracked.item(), 0 if fixed else 1)
            self.assertEqual(torch.equal(before, bn.running_mean), bool(fixed))
            net.eval()
            net.train()  # evaluate() restores the parent mode.
            mode(net, args)
            self.assertEqual(bn.training, not bool(fixed))
            self.assertTrue(net.gaussian_encoder.training)

    def test_raw_validation_history_preserves_precision_and_sample_count(self):
        record = training_helpers()['_record_validation']
        with tempfile.TemporaryDirectory() as tmp:
            history = []
            metrics = {'loc_mean_m':1.23456789, 'loc_median_m':0.9123456}
            loader = SimpleNamespace(dataset=list(range(13)))
            record(history, metrics, 2, loader, Path(tmp))
            saved = json.loads((Path(tmp)/'validation_history.json').read_text())
            self.assertEqual(saved[0]['samples'],13)
            self.assertEqual(saved[0]['metrics'],metrics)
            self.assertEqual(saved[0]['epoch'],2)

    def test_actual_parameter_selection(self):
        for relative, name in [('train_vigor_2DoF.py', 'train'), ('vigor_main/train.py', '_train')]:
            for share in (0, 1):
                with self.subTest(trainer=relative, share=share), tempfile.TemporaryDirectory(prefix='vigor-optimizer-') as tmp:
                    net = TinyNet(share)
                    scope = optimizer_scope(relative, name, net, args_for(share), Path(tmp), small_config())
                    actual = [p for group in scope['optimizer'].param_groups for p in group['params']]
                    modules = [net.gaussian_encoder, net.dpt_sat]
                    if not share:
                        modules.append(net.dpt_grd)
                    expected = {id(p) for module in modules for p in module.parameters()}
                    self.assertEqual({id(p) for p in actual}, expected)
                    self.assertEqual(len(actual), len(expected))
                    self.assertTrue({id(p) for p in net.dino_feat.parameters()}.isdisjoint(expected))

    def test_legacy_gaussian_only_branch(self):
        with tempfile.TemporaryDirectory(prefix='vigor-render-') as tmp:
            net = TinyNet()
            scope = optimizer_scope('train_vigor_2DoF.py', 'train', net,
                                    args_for(supervision='Gaussian'), Path(tmp), small_config())
            actual = {id(p) for group in scope['optimizer'].param_groups for p in group['params']}
            self.assertEqual(actual, {id(p) for p in net.gaussian_encoder.parameters()})
            self.assertIsInstance(scope['optimizer'], optim.Adam)

    def test_real_trainer_uses_loader_length_for_all_steps(self):
        net = TinyNet()
        calls, schedulers = [], []
        batch = tuple(torch.ones(2, 2) for _ in range(8))
        def load(*args, **kwargs):
            calls.append(1)
            return [batch] * 3, []
        def scheduler(*args, **kwargs):
            result = OneCycleLR(*args, **kwargs)
            schedulers.append(result)
            return result
        scope = dict(torch=torch, optim=optim, OneCycleLR=scheduler,
                     config=small_config(), time=time, Path=Path,
                     load_train_and_val=load, evaluate=lambda *a, **k: None,
                     batch_wise_cross_corr=lambda *a, **k: None,
                     Weakly_supervised_loss_w_GPS_error=lambda *a, **k: (net.current_loss, torch.tensor(0.0)))
        scope.update(training_helpers())
        fn = trainer_ast('vigor_main/train.py', '_train')
        exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])), 'vigor_main/train.py', 'exec'), scope)
        with tempfile.TemporaryDirectory(prefix='vigor-schedule-') as tmp:
            scope['_train'](net, args_for(), Path(tmp), torch.device('cpu'))
        self.assertEqual(len(calls), 1)
        self.assertEqual(schedulers[0].total_steps, 6)
        self.assertEqual(schedulers[0].last_epoch, 6)

    def test_forward_does_not_override_encoder_mode(self):
        for relative, cls_name, fn_name in [('models/models_vigor.py', 'ModelVIGOR', 'forward2DoF'),
                                           ('vigor_main/model.py', 'BevSplatVIGOR', 'forward')]:
            tree = ast.parse((ROOT / relative).read_text())
            cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls_name)
            fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == fn_name)
            for node in ast.walk(fn):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                    owner = node.func.value
                    self.assertFalse(isinstance(owner, ast.Attribute) and owner.attr == 'gaussian_encoder'
                                     and node.func.attr in ('train', 'eval'), relative)

    def test_scaled_groups_keep_ratio_through_full_schedule(self):
        for share in (0, 1):
            with tempfile.TemporaryDirectory() as tmp:
                args = args_for(share)
                args.epochs, args.gaussian_lr_scale = 26305, 0.1
                net = TinyNet(share)
                scope = optimizer_scope('vigor_main/train.py', '_train', net, args, Path(tmp), small_config())
                optimizer, scheduler = scope['optimizer'], scope['scheduler']
                self.assertEqual(scheduler.total_steps, 78915)
                self.assertEqual({id(p) for p in optimizer.param_groups[0]['params']},
                                 {id(p) for p in net.gaussian_encoder.parameters()})
                expected_dpt = {id(p) for m in ([net.dpt_sat] if share else [net.dpt_sat, net.dpt_grd]) for p in m.parameters()}
                self.assertEqual({id(p) for p in optimizer.param_groups[1]['params']}, expected_dpt)
                for step in range(scheduler.total_steps):
                    lr = scheduler.get_last_lr()
                    self.assertAlmostEqual(lr[0] / lr[1], 0.1, places=12)
                    optimizer.step()
                    scheduler.step()

    def test_geometry_formula_mask_and_gradient(self):
        from gaussian.encoder_pano import GaussianEncoder
        offsets = torch.tensor([[[[3., 0., 0.], [0., 3., 0.]],
                                 [[9., 9., 9.], [9., 9., 9.]]]], requires_grad=True)
        depth = torch.tensor([[[1.], [float('nan')]]], requires_grad=True)
        prior = GaussianEncoder.offset_geometry_prior(offsets, depth)
        self.assertAlmostEqual(prior.item(), 1/3)
        prior.backward()
        self.assertIsNone(depth.grad)
        self.assertGreater(offsets.grad[0, 0].abs().max().item(), 0)
        self.assertEqual(offsets.grad[0, 1].abs().sum().item(), 0)
        invalid = torch.tensor([[[float('inf')], [0.]]])
        zero = GaussianEncoder.offset_geometry_prior(offsets, invalid)
        self.assertEqual(zero.item(), 0)
        self.assertTrue(zero.requires_grad)

    def test_zero_init_only_position_and_state_contract(self):
        from gaussian.encoder_pano import GaussianEncoder
        class Backbone(nn.Module):
            d_out = 4
            def forward(self, image):
                return image.new_ones(image.shape[0], 4, *image.shape[-2:])
        with patch('gaussian.encoder_pano.BackboneDino', Backbone):
            encoder = GaussianEncoder()
        before = {k:v.clone() for k,v in encoder.state_dict().items()}
        image, feat, conf, depth = torch.rand(1,3,4,8), torch.rand(1,32,4,8), torch.ones(1,1,4,8), torch.ones(1,1,4,8)
        out = encoder(image, feat, conf, depth)
        self.assertTrue(encoder.geometry_prior().requires_grad)
        self.assertEqual(set(before), set(encoder.state_dict()))
        duplicate = encoder(image, feat, conf, depth)
        self.assertTrue(torch.equal(out.means, duplicate.means))
        encoder.zero_initialize_position_offsets()
        position = [0,1,2,11,12,13,22,23,24]
        for key, value in encoder.state_dict().items():
            if key in ('to_gaussians.1.weight', 'to_gaussians.1.bias'):
                self.assertEqual(value[position].abs().sum().item(), 0)
                other = [i for i in range(33) if i not in position]
                self.assertTrue(torch.equal(value[other], before[key][other]))
            else:
                self.assertTrue(torch.equal(value, before[key]))
        encoder(image, feat, conf, depth)
        self.assertEqual(encoder.geometry_prior().item(), 0)

    def test_pilot_exact_steps_then_full_val(self):
        args, net = args_for(), TinyNet()
        args.epochs, args.pilot_steps = 15, 1000
        batch = tuple(torch.ones(2, 2) for _ in range(8))
        val_loader = [object()] * 7
        calls, schedulers = [], []
        def scheduler(*a, **kw):
            result = OneCycleLR(*a, **kw)
            schedulers.append(result)
            return result
        scope = dict(torch=torch, optim=optim, OneCycleLR=scheduler, config=small_config(), time=time,
                     Path=Path, load_train_and_val=lambda *a, **k: ([batch] * 1100, val_loader),
                     evaluate=lambda net, loader, *a, **k: calls.append(loader),
                     batch_wise_cross_corr=lambda *a, **k: None,
                     Weakly_supervised_loss_w_GPS_error=lambda *a, **k: (net.current_loss, torch.tensor(0.)))
        scope.update(training_helpers())
        fn = trainer_ast('vigor_main/train.py', '_train')
        exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])), 'vigor_main/train.py', 'exec'), scope)
        with tempfile.TemporaryDirectory() as tmp:
            scope['_train'](net, args, Path(tmp), torch.device('cpu'))
            self.assertTrue((Path(tmp)/'pilot_step_1000.pth').is_file())
            self.assertFalse((Path(tmp)/'model_0.pth').exists())
        self.assertEqual(schedulers[0].last_epoch, 1000)
        self.assertEqual(schedulers[0].total_steps, 16500)
        self.assertEqual(calls, [val_loader])



def module_digest(module):
    digest = hashlib.sha256()
    for name, tensor in module.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


@unittest.skipUnless(os.environ.get('BEVSPLAT_VIGOR_CUDA_SMOKE') == '1', 'opt-in real CUDA smoke')
class VigorCudaSmoke(unittest.TestCase):
    def test_joint_update(self):
        self.assertEqual(os.environ.get('CUDA_VISIBLE_DEVICES'), '0')
        self.assertTrue(torch.cuda.is_available())
        torch.manual_seed(20261022)
        torch.cuda.set_device(0)
        torch.cuda.reset_peak_memory_stats()
        hub_load = torch.hub.load
        load_weights = torch.hub.load_state_dict_from_url
        def offline_hub(repo, model, *args, **kwargs):
            if kwargs.get('source') != 'local':
                if repo != 'facebookresearch/dino:main':
                    raise RuntimeError('Unapproved uncached hub source: ' + str(repo))
                repo = str(Path(torch.hub.get_dir()) / 'facebookresearch_dino_main')
                self.assertTrue(Path(repo).is_dir())
                kwargs['source'] = 'local'
            return hub_load(repo, model, *args, **kwargs)
        def offline_weights(url, model_dir=None, *args, **kwargs):
            from urllib.parse import urlparse
            filename = kwargs.get('file_name') or Path(urlparse(url).path).name
            cache = Path(model_dir or str(Path(torch.hub.get_dir()) / 'checkpoints')) / filename
            if not cache.is_file():
                raise RuntimeError('Required cached weight missing: ' + str(cache))
            return load_weights(url, model_dir, *args, **kwargs)
        with patch.object(torch.hub, 'load', offline_hub), patch.object(torch.hub, 'load_state_dict_from_url', offline_weights):
            from vigor_main import config
            from vigor_main.train import parse_args, _load_initial, _set_training_mode
            from vigor_main.model import BevSplatVIGOR
            from vigor_main.losses import batch_wise_cross_corr, Weakly_supervised_loss_w_GPS_error
            warm = os.environ.get('BEVSPLAT_VIGOR_WARM_CUDA_SMOKE') == '1'
            flags = ['--init_ckpt', '/data/qiwei/nips25/CVLnet2/ModelsVIGOR/2DoF/same_rot0.0_geo_0.000125_Level1_Channels32_16_4_ConfGrd_Weakly_vigor_0.3_3.0_70_1.25e-4_depth/model_14.pth',
                     '--freeze_gaussian_bn','1','--lr','1e-5'] if warm else [
                     '--geometry_prior_coe','0.1','--zero_init_gaussian_offsets','1']
            args = parse_args(['--batch_size','8','--cuda','0','--gaussian_lr_scale','0.1']+flags)
            device = torch.device('cuda:0')
            net = BevSplatVIGOR(args, device).to(device)
            if warm:
                _load_initial(net, args.init_ckpt)
            else:
                net.gaussian_encoder.zero_initialize_position_offsets()
        with tempfile.TemporaryDirectory(prefix='vigor-cuda-') as tmp:
            scope = optimizer_scope('vigor_main/train.py', '_train', net, args, Path(tmp), config)
        optimizer = scope['optimizer']
        dino_before = module_digest(net.dino_feat)
        bns = [m for m in net.gaussian_encoder.modules() if isinstance(m, nn.BatchNorm2d)]
        self.assertTrue(bns)
        captures = []
        executed_bns = set()
        bn_handles = [m.register_forward_hook(lambda layer, inputs, output: executed_bns.add(id(layer))) for m in bns]
        def hook(module, inputs, output):
            if output.means.requires_grad:
                output.means.retain_grad()
                output.covariances.retain_grad()
            captures.append((module.training, output))
        handle = net.gaussian_encoder.register_forward_hook(hook)
        sat = torch.rand(8, 3, 512, 512, device=device)
        grd = torch.rand(8, 3, 320, 640, device=device)
        depth = torch.full((8, 1, 320, 640), 10.0, device=device)
        zeros = torch.zeros(8, device=device)
        mpp = torch.full((8,), 0.14156, device=device)
        _set_training_mode(net, args)
        bn_buffers_before = {name:t.clone() for name,t in net.gaussian_encoder.named_buffers()
                             if 'running_' in name or 'num_batches_tracked' in name}
        bn_before = {id(m): m.num_batches_tracked.item() for m in bns}
        optimizer.zero_grad()
        outputs = net(sat, grd, depth, grd, mpp, zeros, zeros, zeros)
        self.assertTrue(captures[-1][0])
        self.assertTrue(all(m.training == (not bool(args.freeze_gaussian_bn)) for m in bns))
        used_bns = [m for m in bns if id(m) in executed_bns]
        self.assertTrue(used_bns)
        self.assertTrue(all((m.num_batches_tracked.item() == bn_before[id(m)]) if args.freeze_gaussian_bn
                            else (m.num_batches_tracked.item() > bn_before[id(m)]) for m in used_bns))
        corr = batch_wise_cross_corr(*outputs[:4], args, outputs[4])
        weakly, gps = Weakly_supervised_loss_w_GPS_error(corr, zeros, zeros, args, mpp, args.GPS_error)
        loss = weakly + args.GPS_error_coe * gps + args.geometry_prior_coe * net.gaussian_encoder.geometry_prior()
        self.assertTrue(torch.isfinite(loss).item())
        loss.backward()
        head = net.gaussian_encoder.to_gaussians[-1].weight
        self.assertIsNotNone(head.grad)
        self.assertTrue(torch.isfinite(head.grad).all().item())
        self.assertGreater(head.grad.abs().max().item(), 0)
        geometry = captures[0][1]
        geometry_grads = [t.grad for t in (geometry.means, geometry.covariances) if t.grad is not None]
        self.assertTrue(geometry_grads)
        self.assertTrue(all(torch.isfinite(g).all().item() for g in geometry_grads))
        self.assertTrue(any(g.abs().max().item() > 0 for g in geometry_grads))
        selected = []
        for module in (net.dpt_sat, net.dpt_grd):
            active = [p for p in module.parameters() if p.grad is not None and p.grad.abs().max().item() > 0]
            self.assertTrue(active)
            self.assertTrue(all(torch.isfinite(p.grad).all().item() for p in module.parameters() if p.grad is not None))
            selected.append((active[0], active[0].detach().clone()))
        for group in optimizer.param_groups:
            for parameter in group['params']:
                if parameter.grad is not None:
                    self.assertTrue(torch.isfinite(parameter.grad).all().item(),
                                    'Every gradient selected for update must be finite')
        affine = [p for m in used_bns for p in m.parameters(recurse=False)
                  if p.grad is not None and p.grad.abs().max().item() > 0]
        self.assertTrue(affine)
        selected.append((affine[0], affine[0].detach().clone()))
        before = head.detach().clone()
        optimizer.step()
        if args.freeze_gaussian_bn:
            self.assertTrue(all(torch.equal(value, bn_buffers_before[name])
                                for name,value in net.gaussian_encoder.named_buffers() if name in bn_buffers_before))
        self.assertFalse(torch.equal(head.detach(), before))
        self.assertTrue(all(not torch.equal(p.detach(), value) for p, value in selected))
        torch.cuda.synchronize()
        free, total = torch.cuda.mem_get_info()
        self.assertGreaterEqual(free, 1024 ** 3, 'Shared GPU0 requires at least1GiB free headroom')
        counters = [m.num_batches_tracked.item() for m in bns]
        net.eval()
        with torch.no_grad():
            net(sat, grd, depth, grd, mpp, zeros, zeros, zeros)
        self.assertFalse(captures[-1][0])
        self.assertTrue(all(not m.training for m in bns))
        self.assertEqual(counters, [m.num_batches_tracked.item() for m in bns])
        _set_training_mode(net, args)
        self.assertTrue(net.gaussian_encoder.training)
        self.assertTrue(all(m.training == (not bool(args.freeze_gaussian_bn)) for m in bns))
        self.assertEqual(dino_before, module_digest(net.dino_feat))
        self.assertTrue(all(p.grad is None for p in net.dino_feat.parameters()))
        handle.remove()
        for bn_handle in bn_handles:
            bn_handle.remove()
        print('VIGOR_CUDA_SMOKE=' + json.dumps({'batch_size':8, 'sat_shape':list(sat.shape),
              'grd_shape':list(grd.shape), 'depth_shape':list(depth.shape), 'dtype':str(sat.dtype),
              'loss':loss.item(), 'gaussian_head_grad_max':head.grad.abs().max().item(),
              'geometry_grad_max':[g.abs().max().item() for g in geometry_grads],
              'peak_allocated_MiB':torch.cuda.max_memory_allocated()/1024**2,
              'peak_reserved_MiB':torch.cuda.max_memory_reserved()/1024**2,
              'free_after_step_MiB':free/1024**2, 'total_MiB':total/1024**2,
              'dino_unchanged':True, 'gaussian_updated':True, 'dpt_updated':True, 'bn_modes_passed':True,
              'warm_start':warm, 'fixed_gaussian_bn':bool(args.freeze_gaussian_bn), 'bn_affine_updated':True}), flush=True)


if __name__ == '__main__':
    unittest.main()
