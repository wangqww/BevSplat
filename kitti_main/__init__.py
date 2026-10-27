"""Clean entry point for BevSplat's KITTI main experiment.

Historical KITTI wrapper, not newly verified as reproducing paper tables
in the current VIGOR E2 release. It wraps the algorithm
modules that live elsewhere in the repo:

    gaussian/encoder_feat_nips.py   -- per-pixel Gaussian primitive encoder
    vis_gaussian_feat.py            -- CUDA orthographic BEV splat
    models/dino_fit.py              -- DINOv2 backbone (with FiT weights)
    models/dpt_single.py            -- DPT head producing 32-d feature + confidence
    models/VGGW.py                  -- VGGUnet (rotation pre-head)
    models/swin_transformer.py      -- TransOptimizerG2SP_V1 (rotation pre-head)
    dataLoader/KITTI_dataset.py     -- KITTI sat/grd pair loader

Reproduction recipes live in ``kitti_main/README.md``.
"""
