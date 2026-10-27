# Repository development notes

The supported release workflow is documented in README.md and docs/. Current VIGOR E2 explicitly selects corrected behavior and crop decoding; compatibility defaults can differ. Do not infer runtime settings from a checkpoint filename/state dictionary.

- Entry/optimizer/schedule: vigor_main/train.py; model: vigor_main/model.py.
- Loss/correlation: vigor_main/losses.py; native metrics/MAT: vigor_main/eval.py.
- Ordered data splits: vigor_main/data.py and dataLoader/Vigor_dataset_gs.py.
- Gaussian encoder: gaussian/encoder_pano.py; renderer: vis_gaussian_pano.py and pano_feature_gaussian/.
- Public code/cache overrides: models/public_pretrained.py, models/dino_fit.py.
- Portable pinned preprocessing bridge: scripts/prepare_vigor_depth.py.
- Public CPU regressions: tests/test_vigor_*.py; GPU/private-controller tests may skip.

Preserve scientific objective, labels, split order, decoder origin and native float32 strict-threshold metrics when making packaging changes. Model-only snapshots are not exact optimizer/RNG resume checkpoints. Do not claim paper reproduction from current corrected E2 results. KITTI/sequence/Oxford and old scratch scripts remain historical, outside the supported VIGOR recipe. Retain third-party rasterizer licenses.
