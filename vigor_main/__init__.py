"""Clean entry point for BevSplat's VIGOR 2DoF main experiment.

Provides current VIGOR Same/Cross and GPS-coefficient experiments.
Current corrected E2 results are separate from historical paper rows. Same shape as ``kitti_main/`` but adapted to:

    - panoramic ground images (equirectangular sphere lift, not perspective rays)
    - the ``pano_gaussian_feat`` CUDA splatter (not ``feat_gaussian``)
    - two DPT heads (``dpt_sat`` / ``dpt_grd``) instead of one shared
    - 2-DoF pose (translation only; rotation pre-head is skipped because the
      VIGOR pipeline conditions on GT rotation through ``self.sat2grd_uv``)

Algorithm modules reused verbatim:

    gaussian/encoder_pano.py     -- per-pixel Gaussian primitive encoder (panorama)
    vis_gaussian_pano.py         -- CUDA orthographic BEV splat
    backbone/backbone_pano.py    -- DINO-pretrained backbone for the encoder
    models/dino_fit.py           -- DINOv2 + FiT feature trunk
    models/dpt_single.py         -- DPT head producing 32-d feature + confidence
    models/VGGW.py               -- VGGUnet (kept instantiated for checkpoint compat)
    dataLoader/Vigor_dataset_gs.py -- VIGOR sat/grd/depth/oxts loader

Reproduction recipes live in ``vigor_main/README.md``.
"""
