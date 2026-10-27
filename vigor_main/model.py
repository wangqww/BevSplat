"""``BevSplatVIGOR`` — the Stage-1 2DoF model for VIGOR.

Focused rewrite of ``ModelVIGOR.forward2DoF`` (``models/models_vigor.py``
lines 234–358). The legacy class also carries a half-finished
"forward_project" path and a Gaussian-render supervision branch — both
are dropped here. Only the Stage-1 ``Weakly``-supervised forward is
preserved.

Submodule names match the legacy ``ModelVIGOR`` exactly so existing
``.pth`` state names need no key remapping. Public evaluation validates and
loads the complete state strictly; runtime behavior still needs explicit flags.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from models.VGGW import VGGUnet
from models.dino_fit import DINO
from models.dpt_single import DPT
from gaussian.encoder_pano import GaussianEncoder
from vis_gaussian_pano import render_projections

from .config import BEV_CROP_RATIO, BEV_HEIGHT_M, BEV_RES, BEV_WIDTH_M, GRD_H, GRD_W, SAT_H, SAT_W


class BevSplatVIGOR(nn.Module):
    """Stage-1 2DoF BevSplat model for VIGOR.

    Args
    ----
    args
        Namespace carrying ``level`` (e.g. ``"1"``), ``channels``
        (``"32_16_4"``), ``share`` (0/1), ``rotation_range`` (0 for
        the main 2DoF protocol), ``N_iters``, ``ConfGrd``, ``ConfSat``,
        ``grd_res``, ``area``.
    device
        Target device for module construction (the ``camera_k`` buffer
        and gaussian encoder live here on creation).
    """

    def __init__(self, args, device):
        super().__init__()
        self.args = args
        self.model_behavior = getattr(args, "model_behavior", "legacy")
        if self.model_behavior not in ("legacy", "corrected"):
            raise ValueError("Unknown model_behavior: " + str(self.model_behavior))
        override = getattr(args, "confidence_mode", "inherit")
        if override not in ("inherit", "legacy", "corrected"):
            raise ValueError("Unknown confidence_mode: " + str(override))
        self.confidence_mode = self.model_behavior if override == "inherit" else override
        self.device = device
        self.share = args.share
        self.grd_res = args.grd_res
        self.level = sorted(int(item) for item in args.level.split("_"))[0]
        self.N_iters = args.N_iters
        self.channels = [int(item) for item in args.channels.split("_")]

        # Per-pixel Gaussian primitive encoder for the panoramic grd image.
        self.gaussian_encoder = GaussianEncoder(area=args.area)
        limits = (getattr(args, "gaussian_offset_max", 3.0),
                  getattr(args, "gaussian_scale_max", 0.3),
                  getattr(args, "gaussian_opacity_max", 1.0))
        if limits != (3.0, 0.3, 1.0):
            self.gaussian_encoder.configure_ranges(*limits)

        # DINOv2 trunk + two DPT heads (32-d feature + confidence).
        # ``share=1`` reuses dpt_sat for both branches but the dpt_grd
        # module is still instantiated for checkpoint compat.
        self.dino_feat = DINO()
        self.dpt_sat = DPT(self.dino_feat.feat_dim, confidence_mode=self.confidence_mode)
        self.dpt_grd = DPT(self.dino_feat.feat_dim, confidence_mode=self.confidence_mode)

        # Kept instantiated to match legacy ckpt keys; not used in the
        # Stage-1 ``Weakly``-supervised forward.
        self.FeatureForT = VGGUnet(self.level, self.channels)
        self.SatFeatureNet = VGGUnet(self.level, self.channels)
        self.GrdFeatureNet = VGGUnet(self.level, self.channels)

        self.global_step = 0
        self.camera_k = torch.tensor(
            [[0.5000, 0.0000, 0.5000],
             [0.0000, 0.5000, 0.5000],
             [0.0000, 0.0000, 1.0000]],
            device=device,
        ).unsqueeze(0)

    # ------------------------------------------------------------------
    # Stage-1 2DoF forward
    # ------------------------------------------------------------------

    def forward(
        self,
        sat,
        grd,
        depth_imgs,
        grd_ori,            # unused; kept in signature for dataloader compat
        meter_per_pixel,
        gt_rot=None,
        gt_shift_u=None,    # unused at forward time; consumed by the loss
        gt_shift_v=None,
        stage=None,
        loop=None,
        save_dir=None,
    ):
        """Stage-1 forward — produces the dicts consumed by ``losses.py``.

        Returns
        -------
        sat_feat_dict, sat_conf_dict, g2s_feat_dict, g2s_conf_dict, sat_uncer_dict
        """
        del grd_ori, gt_shift_u, gt_shift_v, stage, loop, save_dir
        b = sat.shape[0]

        # --- Resize satellite to DPT input resolution -------------------
        sat_resized = F.interpolate(sat, (SAT_H, SAT_W), mode="bilinear", align_corners=True)

        # --- DINOv2 + DPT features --------------------------------------
        with torch.no_grad():
            sat_feats = self.dino_feat(sat)         # full-resolution input
            grd_feats = self.dino_feat(grd)
            if isinstance(sat_feats, (tuple, list)):
                sat_feats = [_f.detach() for _f in sat_feats]
            if isinstance(grd_feats, (tuple, list)):
                grd_feats = [_f.detach() for _f in grd_feats]

        if self.share:
            sat_feat, sat_conf = self.dpt_sat(sat_feats)
            grd_feat, grd_conf = self.dpt_sat(grd_feats)
        else:
            sat_feat, sat_conf = self.dpt_sat(sat_feats)
            grd_feat, grd_conf = self.dpt_grd(grd_feats)

        sat_feat_dict = {self.level: sat_feat}
        sat_conf_dict = {self.level: sat_conf}
        grd_feat_dict = {self.level: grd_feat}
        grd_conf_dict = {self.level: grd_conf}

        # --- Lift the panorama into 3D Gaussians ------------------------
        gs_grd = F.interpolate(grd, (GRD_H, GRD_W), mode="bilinear", align_corners=True)
        gs_depth_img = F.interpolate(depth_imgs, (GRD_H, GRD_W), mode="bilinear", align_corners=True)

        grd_gaussian = self.gaussian_encoder(
            gs_grd,
            grd_feat_dict[self.level],
            grd_conf_dict[self.level],
            gs_depth_img,
        )

        # --- Render the Gaussian cloud into BEV -------------------------
        # 2DoF means the heading comes from the GT pose (no rotation
        # pre-head). The "+90°" mirrors the equirectangular convention
        # used inside render_projections.
        if self.args.rotation_range == 0:
            heading = torch.ones_like(gt_rot.unsqueeze(-1), device=gt_rot.device) * 90
            rot_range = 1
        else:
            heading = torch.ones_like(gt_rot.unsqueeze(-1), device=gt_rot.device) * 90 / self.args.rotation_range
            heading = heading + gt_rot.unsqueeze(-1)
            rot_range = self.args.rotation_range

        _, bev_feat, bev_conf, _ = render_projections(
            grd_gaussian,
            BEV_RES,
            heading=heading,
            rot_range=rot_range,
            width=BEV_WIDTH_M,
            height=BEV_HEIGHT_M,
            use_predicted_opacity=(self.model_behavior == "corrected"),
        )

        # --- Crop BEV to the pose-search window -------------------------
        if getattr(self.args, "translation_origin", "legacy") == "crop":
            if bev_feat.shape[-2:] != sat_feat_dict[self.level].shape[-2:]:
                raise ValueError("Crop decoding requires identical full satellite and BEV grids")
        A = sat_feat_dict[self.level].shape[-1]
        crop = int(A * BEV_CROP_RATIO)
        g2s_feat = TF.center_crop(bev_feat, [crop, crop])
        g2s_conf = TF.center_crop(bev_conf, [crop, crop])
        g2s_feat_dict = {self.level: g2s_feat}
        g2s_conf_dict = {self.level: g2s_conf}

        sat_uncer_dict = {lvl: None for lvl in range(3)}
        return sat_feat_dict, sat_conf_dict, g2s_feat_dict, g2s_conf_dict, sat_uncer_dict
