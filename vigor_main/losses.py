"""Loss functions and correlation primitives for the VIGOR main experiment.

Math copied verbatim from ``models/models_vigor.py``:

    batch_wise_cross_corr                    L364  ->  used in training
    Weakly_supervised_loss_w_GPS_error       L473  ->  paper Eq. 1 (L_Weakly + L_GPS)
    corr_for_translation                     L549  ->  used at eval time

Shape conventions are the same as KITTI; the only practical differences
are:

    * the GT-shift to grid-cell index mapping uses
      ``gt_shift_* * 512 / np.power(2, 3 - level) / 4`` instead of the
      KITTI ``gt_shift_* * shift_range_*`` formula — VIGOR shifts are
      stored in normalized image-fraction units rather than meters;
    * the GPS-error radius is per-sample (``meters_per_pixel`` is a
      ``[B]`` tensor with each sample's city scale);
    * a single-sample batch (``M == 1``) falls back to the unnormalized
      sum (``(N-1)`` would be zero).

``corr = 2 - 2 * normalized_cosine`` everywhere, so smaller = more
similar (consistent with the KITTI version).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF


# ---------------------------------------------------------------------------
# Training-time correlation
# ---------------------------------------------------------------------------

def batch_wise_cross_corr(
    sat_feat_dict,
    sat_conf_dict,
    g2s_feat_dict,
    g2s_conf_dict,
    args,
    masks=None,
):
    """Per-pyramid-level batch-wise correlation. See KITTI's losses.py for
    the shape contract — VIGOR's version is identical aside from the
    ConfGrd=0 branch using a binary ``masks[level]`` for the L2 denom."""
    levels = sorted(int(item) for item in args.level.split("_"))
    corr_maps = {}
    for level in levels:
        sat_feat = sat_feat_dict[level]
        sat_conf = sat_conf_dict[level]
        g2s_feat = g2s_feat_dict[level]
        g2s_conf = g2s_conf_dict[level]

        B, C, crop_H, crop_W = g2s_feat.shape

        if args.ConfGrd > 0:
            if args.ConfSat > 0:
                signal = (sat_feat * sat_conf.pow(2)).repeat(1, B, 1, 1)
                kernel = g2s_feat * g2s_conf.pow(2)
                corr = F.conv2d(signal, kernel, groups=B)

                denominator_sat = []
                sat_feat_conf_pow = (sat_feat * sat_conf).pow(2)
                g2s_conf_pow = g2s_conf.pow(2)
                for i in range(B):
                    denom_sat = torch.sum(
                        F.conv2d(sat_feat_conf_pow[i, :, None, :, :], g2s_conf_pow),
                        dim=0,
                    )
                    denominator_sat.append(denom_sat)
                denominator_sat = torch.sqrt(torch.stack(denominator_sat, dim=0))

                denominator_grd = []
                sat_conf_pow = sat_conf.pow(2)
                g2s_feat_conf_pow = (g2s_feat * g2s_conf).pow(2)
                for i in range(B):
                    denom_grd = torch.sum(
                        F.conv2d(sat_conf_pow[i : i + 1, :, :, :].repeat(1, C, 1, 1), g2s_feat_conf_pow),
                        dim=1,
                    )
                    denominator_grd.append(denom_grd)
                denominator_grd = torch.sqrt(torch.stack(denominator_grd, dim=0))

            else:
                signal = sat_feat.repeat(1, B, 1, 1)
                kernel = g2s_feat * g2s_conf.pow(2)
                corr = F.conv2d(signal, kernel, groups=B)

                denominator_sat = []
                sat_feat_pow = sat_feat.pow(2)
                g2s_conf_pow = g2s_conf.pow(2)
                for i in range(B):
                    denom_sat = torch.sum(
                        F.conv2d(sat_feat_pow[i, :, None, :, :], g2s_conf_pow),
                        dim=0,
                    )
                    denominator_sat.append(denom_sat)
                denominator_sat = torch.sqrt(torch.stack(denominator_sat, dim=0))

                denom_grd = torch.linalg.norm((g2s_feat * g2s_conf).reshape(B, -1), dim=-1)
                shape = denominator_sat.shape
                denominator_grd = denom_grd[None, :, None, None].repeat(shape[0], 1, shape[2], shape[3])

        else:
            mask = TF.center_crop(masks[level].permute(0, 3, 1, 2), [crop_H, crop_W]).float()
            signal = sat_feat.repeat(1, B, 1, 1)
            kernel = g2s_feat
            corr = F.conv2d(signal, kernel, groups=B)

            l2_norm_kernel = mask.repeat(1, C, 1, 1)
            sat_feat_squared_sum = F.conv2d(signal.pow(2), l2_norm_kernel, stride=1, padding=0, groups=B)
            denominator_sat = torch.sqrt(sat_feat_squared_sum + 1e-8)

            denom_grd = torch.linalg.norm(g2s_feat.reshape(B, -1), dim=-1)
            shape = denominator_sat.shape
            denominator_grd = denom_grd[None, :, None, None].repeat(shape[0], 1, shape[2], shape[3])

        denominator = torch.maximum(denominator_sat * denominator_grd, torch.full_like(denominator_sat, 1e-6))
        corr_maps[level] = 2 - 2 * corr / denominator

    return corr_maps


# ---------------------------------------------------------------------------
# Loss: L_Weakly + L_GPS
# ---------------------------------------------------------------------------

def Weakly_supervised_loss_w_GPS_error(
    corr_maps,
    gt_shift_u,
    gt_shift_v,
    args,
    meters_per_pixel,
    GPS_error: float = 5,
):
    """Returns ``(L_Weakly, L_GPS)``.

    Note the GT-shift to grid-index conversion: VIGOR's ``gt_shift_*`` are
    normalized so that ``gt_shift * 512 / 2**(3-level) / 4`` maps into
    the correlation map's coordinate system. ``meters_per_pixel`` is a
    per-sample ``[B]`` tensor because VIGOR cities have different scales.
    """
    matching_losses = []
    GPS_error_losses = []

    levels = [int(item) for item in args.level.split("_")]
    for level in levels:
        corr = corr_maps[level]
        M, N, H, W = corr.shape
        assert M == N

        dis = torch.min(corr.reshape(M, N, -1), dim=-1)[0]
        pos = torch.diagonal(dis)
        pos_neg = pos.reshape(-1, 1) - dis
        if M == 1:
            loss = torch.sum(torch.log(1 + torch.exp(pos_neg * 10)))
        else:
            loss = torch.sum(torch.log(1 + torch.exp(pos_neg * 10))) / (M * (N - 1))
        matching_losses.append(loss)

        scale = 512 / np.power(2, 3 - level) / 4
        w = torch.round(W / 2 - 0.5 + gt_shift_u * scale).long()
        h = torch.round(H / 2 - 0.5 + gt_shift_v * scale).long()
        radius = torch.ceil(GPS_error / (meters_per_pixel * np.power(2, 3 - level))).long()

        GPS_dis = []
        for b in range(M):
            start_h = torch.max(torch.tensor(0).long(), h[b] - radius[b])
            end_h = torch.min(torch.tensor(corr.shape[2]).long(), h[b] + radius[b])
            start_w = torch.max(torch.tensor(0).long(), w[b] - radius[b])
            end_w = torch.min(torch.tensor(corr.shape[3]).long(), w[b] + radius[b])
            GPS_dis.append(torch.min(corr[b, b, start_h:end_h, start_w:end_w]))
        GPS_error_losses.append(torch.abs(torch.stack(GPS_dis) - pos))

    return (
        torch.mean(torch.stack(matching_losses, dim=0)),
        torch.mean(torch.stack(GPS_error_losses, dim=0)),
    )


# ---------------------------------------------------------------------------
# Inference-time correlation
# ---------------------------------------------------------------------------

def _translation_origin(length, mode):
    """Feature-cell origin for equal full grids and torchvision center_crop."""
    if mode == "legacy":
        return length / 2
    if mode == "crop":
        return float(int(round((length - 1) / 2.0)))
    raise ValueError("Unknown translation_origin: " + str(mode))


def corr_for_translation(
    sat_feat_dict,
    sat_conf_dict,
    g2s_feat_dict,
    g2s_conf_dict,
    args,
    sat_uncer_dict=None,
):
    """Eval-time correlation — returns ``(pred_u, pred_v, corr)``.

    ``pred_u`` and ``pred_v`` are in *correlation-pixel* units; the
    caller multiplies by ``meter_per_pixel`` to get metric shifts.
    """
    level = max(int(item) for item in args.level.split("_"))

    sat_feat = sat_feat_dict[level]
    sat_conf = sat_conf_dict[level]
    g2s_feat = g2s_feat_dict[level]
    g2s_conf = g2s_conf_dict[level]

    B, C, crop_H, crop_W = g2s_feat.shape
    A = sat_feat.shape[2]

    if args.ConfGrd > 0:
        if args.ConfSat > 0:
            signal = (sat_feat * sat_conf.pow(2)).reshape(1, -1, A, A)
            kernel = g2s_feat * g2s_conf.pow(2)
            corr = F.conv2d(signal, kernel, groups=B)[0]

            sat_feat_conf_pow = (sat_feat * sat_conf).pow(2).transpose(0, 1)
            g2s_conf_pow = g2s_conf.pow(2)
            denominator_sat = F.conv2d(sat_feat_conf_pow, g2s_conf_pow, groups=B).transpose(0, 1)
            denominator_sat = torch.sqrt(torch.sum(denominator_sat, dim=1))

            sat_conf_pow = sat_conf.pow(2).repeat(1, C, 1, 1).reshape(1, -1, A, A)
            g2s_feat_conf_pow = (g2s_feat * g2s_conf).pow(2)
            denominator_grd = F.conv2d(sat_conf_pow, g2s_feat_conf_pow, groups=B)[0]
            denominator_grd = torch.sqrt(denominator_grd)
        else:
            signal = sat_feat.reshape(1, -1, A, A)
            kernel = g2s_feat * g2s_conf.pow(2)
            corr = F.conv2d(signal, kernel, groups=B)[0]

            sat_feat_pow = sat_feat.pow(2).transpose(0, 1)
            g2s_conf_pow = g2s_conf.pow(2)
            denominator_sat = F.conv2d(sat_feat_pow, g2s_conf_pow, groups=B).transpose(0, 1)
            denominator_sat = torch.sqrt(torch.sum(denominator_sat, dim=1))

            denom_grd = torch.linalg.norm((g2s_feat * g2s_conf).reshape(B, -1), dim=-1)
            shape = denominator_sat.shape
            denominator_grd = denom_grd[:, None, None].repeat(1, shape[1], shape[2])
    else:
        signal = sat_feat.reshape(1, -1, A, A)
        kernel = g2s_feat
        corr = F.conv2d(signal, kernel, groups=B)[0]

        denominator_sat = F.avg_pool2d(sat_feat.pow(2), (crop_H, crop_W), stride=1, divisor_override=1)
        denominator_sat = torch.sqrt(torch.sum(denominator_sat, dim=1))

        denom_grd = torch.linalg.norm(g2s_feat.reshape(B, -1), dim=-1)
        shape = denominator_sat.shape
        denominator_grd = denom_grd[:, None, None].repeat(1, shape[1], shape[2])

    denominator = denominator_sat * denominator_grd
    if getattr(args, "use_uncertainty", 0):
        denominator = denominator * TF.center_crop(sat_uncer_dict[level], [corr.shape[1], corr.shape[2]])[:, 0]
    denominator = torch.maximum(denominator, torch.full_like(denominator, 1e-6))
    corr = corr / denominator

    B, corr_H, corr_W = corr.shape
    max_index = torch.argmax(corr.reshape(B, -1), dim=1)
    origin_mode = getattr(args, "translation_origin", "legacy")
    pred_u = max_index % corr_W - _translation_origin(corr_W, origin_mode)
    pred_v = max_index // corr_W - _translation_origin(corr_H, origin_mode)

    return pred_u * np.power(2, 3 - level), pred_v * np.power(2, 3 - level), corr
