"""Centralized constants for the VIGOR 2DoF main experiment.

Every magic number that was previously scattered across
``train_vigor_2DoF.py`` and ``models/models_vigor.py`` lives here.
Values come from the *existing code*, not the paper. Where the paper
and code disagree, a comment notes which value we are using.
"""

import os
from pathlib import Path

# ----------------------------------------------------------------------
# Filesystem
# ----------------------------------------------------------------------

# VIGOR dataset root. Set in dataLoader/Vigor_dataset_gs.py:15.
VIGOR_ROOT = Path(os.environ.get("VIGOR_ROOT", "/data/dataset/VIGOR")).expanduser()

# Where checkpoints land. Mirrors train_vigor_2DoF.py:667.
CKPT_ROOT = Path(os.environ.get("VIGOR_CKPT_ROOT", "/data/qiwei/nips25/CVLnet2/ModelsVIGOR/2DoF")).expanduser()

# ----------------------------------------------------------------------
# Image / BEV geometry
# ----------------------------------------------------------------------

# Satellite resize before DINOv2 / DPT. Source: models_vigor.py:241.
SAT_H = 128
SAT_W = 128

# Panoramic ground image size fed to the Gaussian encoder. The legacy
# code hardcodes ``grd_res = 80`` *inside* forward2DoF (line 236),
# overriding the ``args.grd_res = 40`` from argparse. We keep the
# active value.
GRD_RES = 80           # height of the equirectangular grd tile
GRD_H = GRD_RES        # 80
GRD_W = GRD_RES * 2    # 160 — equirectangular is always 2:1

# BEV extent in meters and pixel resolution. Source: models_vigor.py:299.
BEV_WIDTH_M = 70.0
BEV_HEIGHT_M = 70.0
BEV_RES = (128, 128)

# Cropping factor applied to BEV features before correlation.
# Source: models_vigor.py:345-346 ``crop = int(A * 0.4)``.
BEV_CROP_RATIO = 0.4

# Gaussians per equirectangular pixel. Source: encoder_pano.py:62.
GAUSSIANS_PER_PIXEL = 3

# ----------------------------------------------------------------------
# Feature pyramid / model knobs (single-value flags from legacy argparse)
# ----------------------------------------------------------------------

LEVEL = 1
LEVEL_STR = "1"
CHANNELS_STR = "32_16_4"

SHARE_FEATURE_NET = 0     # VIGOR default differs from KITTI: separate dpt_grd / dpt_sat
CONF_GRD = 1
CONF_SAT = 0
N_ITERS = 1
PROJ = "geo"
OPTIMIZER_KIND = "TransV1"
USE_UNCERTAINTY = 0

SUPERVISION = "Weakly"    # the main path; the legacy "Gaussian" branch is for the render-loss pretrain

# ----------------------------------------------------------------------
# Training hyperparameters
# ----------------------------------------------------------------------

LR = 1.25e-4              # default for Same-Area; Cross-Area uses 1e-4 (see README)
WEIGHT_DECAY = 5e-3       # NOTE: paper says 1e-3; code says 5e-3; we follow code
ADAM_EPS = 1e-8
BATCH_SIZE = 8
EPOCHS = 15
ONECYCLE_STEPS_PER_EPOCH_BASE = 5260   # VIGOR same-area train batches at BS=8
ONECYCLE_PCT_START = 0.01
ONECYCLE_ANNEAL = "cos"

# ----------------------------------------------------------------------
# Loss
# ----------------------------------------------------------------------

GPS_ERROR_COE_LAMBDA1 = (0.0, 1.0)
GPS_ERROR_RADIUS_M = 5.0

# Pose perturbation — VIGOR is 2DoF: rotation_range = 0 means GT heading.
ROTATION_RANGE = 0.0

# Data subsample fraction; legacy ``--amount`` default is 1.0.
DATA_AMOUNT = 1.0

# ----------------------------------------------------------------------
# Evaluation thresholds (paper Table 2 columns)
# ----------------------------------------------------------------------

DIST_THRESHOLDS_M = (1, 3, 5)


def save_dir(
    *,
    area: str = "same",
    rotation_range: float = ROTATION_RANGE,
    proj: str = PROJ,
    lr: float = LR,
    level_str: str = LEVEL_STR,
    channels_str: str = CHANNELS_STR,
    conf_grd: int = CONF_GRD,
    conf_sat: int = CONF_SAT,
    gps_error: float = GPS_ERROR_RADIUS_M,
    gps_error_coe: float = 0.0,
    share: int = SHARE_FEATURE_NET,
    supervision: str = SUPERVISION,
    name: str = "",
) -> Path:
    """Reproduce the save_path layout from train_vigor_2DoF.py:666-692.

    Keeping the same directory naming means existing checkpoints can be
    resumed by the refactored trainer without renaming any files.
    """
    parts = (
        f"{area}_rot{rotation_range}"
        f"_{proj}_{lr}"
        f"_Level{level_str}_Channels{channels_str}"
    )
    path = CKPT_ROOT / parts
    if conf_grd:
        path = path.parent / (path.name + "_ConfGrd")
    if conf_sat:
        path = path.parent / (path.name + "_ConfSat")
    if gps_error_coe > 0:
        path = path.parent / (path.name + f"_GPSerror{gps_error}_Coe{gps_error_coe}")
    if share:
        path = path.parent / (path.name + "_Share")
    path = path.parent / (path.name + f"_{supervision}")
    if name:
        path = path.parent / (path.name + f"_{name}")
    return path
