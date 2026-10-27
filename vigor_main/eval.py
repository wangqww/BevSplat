"""VIGOR evaluation — one function replacing the legacy ``test`` / ``val``.

Both legacy functions were near-duplicates; ``val`` also carried a
130-line matplotlib visualization block and a "save best model" side
effect. This module collapses them into a single
``evaluate(loader, ...)`` call that reports native localization metrics.

Metric definitions (per the legacy script):

    distance = sqrt((pred_u - gt_u)² + (pred_v - gt_v)²)
    where pred_u/v are in METERS (= corr-pixel * meter_per_pixel)
    and gt_u/v also in METERS (= gt_shift * meter_per_pixel * 512/4).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Sequence

import numpy as np
import scipy.io as scio
import torch

from .config import DIST_THRESHOLDS_M
from .losses import corr_for_translation


@torch.no_grad()
def evaluate(
    net,
    loader,
    args,
    save_path: str | Path,
    epoch: int,
    split_name: str,
    *,
    device: torch.device | str = "cuda",
    dist_thresholds: Sequence[int] = DIST_THRESHOLDS_M,
):
    """Evaluate ``net`` on one VIGOR loader.

    Parameters
    ----------
    net
        ``BevSplatVIGOR`` (or the legacy ``ModelVIGOR``) instance.
    loader
        DataLoader from ``vigor_main.data.load_test``.
    args
        Namespace forwarded to ``corr_for_translation`` (needs
        ``level``, ``ConfGrd``, ``ConfSat``, ``use_uncertainty``).
    save_path
        Directory where ``<split>_results.txt`` and ``<split>_result.mat``
        are written.
    epoch
        Recorded in the txt header.
    split_name
        ``"same"`` or ``"cross"`` (used in filenames and stdout).

    Returns
    -------
    dict
        Flat metric dict; keys match the Table-2 columns.
    """
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    net.eval()

    pred_us, pred_vs, gt_us, gt_vs = [], [], [], []
    n_batches = len(loader)
    print(f"[{split_name}] batch_size={args.batch_size}, num_batches={n_batches}")
    t0 = time.time()

    for i, batch in enumerate(loader):
        # train_loader and val/test_loader both return 8 items in
        # Vigor_dataset_gs (see data.py docstring).
        (
            grd,
            sat,
            depth_imgs,
            grd_ori,
            gt_shift_u,
            gt_shift_v,
            gt_rot,
            meter_per_pixel,
        ) = (item.to(device) for item in batch)

        sat_feat_d, sat_conf_d, g2s_feat_d, g2s_conf_d, sat_uncer_d = net(
            sat, grd, depth_imgs, grd_ori, meter_per_pixel,
            gt_rot, gt_shift_u, gt_shift_v,
        )
        pred_u, pred_v, _ = corr_for_translation(
            sat_feat_d, sat_conf_d, g2s_feat_d, g2s_conf_d, args, sat_uncer_d
        )

        # Convert correlation-pixel offsets to meters.
        pred_u = pred_u * meter_per_pixel
        pred_v = pred_v * meter_per_pixel
        # GT shifts are stored normalized; multiply by ``meter_per_pixel * 512/4``
        # to land in the same metric frame (the satellite tile is 512px wide and
        # the dataloader applies a /4 scale to map shift fractions to pixels).
        gt_u_m = gt_shift_u * meter_per_pixel * 512 / 4
        gt_v_m = gt_shift_v * meter_per_pixel * 512 / 4

        pred_us.append(pred_u.cpu().numpy())
        pred_vs.append(pred_v.cpu().numpy())
        gt_us.append(gt_u_m.cpu().numpy())
        gt_vs.append(gt_v_m.cpu().numpy())

        if i % 20 == 0:
            print(f"[{split_name}] batch {i}/{n_batches}")

    duration_per_image = (time.time() - t0) / max(n_batches * args.batch_size, 1)
    pred_us = np.concatenate(pred_us)
    pred_vs = np.concatenate(pred_vs)
    gt_us = np.concatenate(gt_us)
    gt_vs = np.concatenate(gt_vs)

    scio.savemat(
        save_path / f"{split_name}_result.mat",
        {"pred_us": pred_us, "pred_vs": pred_vs, "gt_us": gt_us, "gt_vs": gt_vs},
    )

    distance = np.sqrt((pred_us - gt_us) ** 2 + (pred_vs - gt_vs) ** 2)
    init_dis = np.sqrt(gt_us ** 2 + gt_vs ** 2)

    metrics = {
        "loc_mean_m": float(np.mean(distance)),
        "loc_median_m": float(np.median(distance)),
        "init_mean_m": float(np.mean(init_dis)),
        "init_median_m": float(np.median(init_dis)),
    }
    for t in dist_thresholds:
        metrics[f"dist_d={t}m_%"] = float(np.mean(distance < t) * 100)

    _report(
        save_path / f"{split_name}_results.txt",
        metrics,
        split_name,
        epoch,
        duration_per_image,
        dist_thresholds,
    )

    net.train()
    return metrics


def _report(
    out_path: Path,
    metrics: dict,
    split_name: str,
    epoch: int,
    duration_per_image: float,
    dist_thresholds: Sequence[int],
):
    lines = [
        "====================================",
        f"  VIGOR {split_name}   EPOCH: {epoch}",
        f"  Time per image (s): {duration_per_image:.4f}",
        "------------------------------------",
        f"  Init     mean (m): {metrics['init_mean_m']:.3f}    median (m): {metrics['init_median_m']:.3f}",
        f"  Predicted mean (m): {metrics['loc_mean_m']:.3f}    median (m): {metrics['loc_median_m']:.3f}",
        "------------------------------------",
    ]
    for t in dist_thresholds:
        lines.append(f"  Localization within d={t}m : {metrics[f'dist_d={t}m_%']:.2f} %")
    lines.append("====================================")
    body = "\n".join(lines) + "\n"
    print(body)
    with out_path.open("a") as f:
        f.write(body)
