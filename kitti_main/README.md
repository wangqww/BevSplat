# Historical KITTI entry point

This package is retained for earlier KITTI experiments. The current release's executable recipe and validation concern [corrected VIGOR E2](../README.md); KITTI paper tables were not freshly reproduced.

KITTI/sequence/Oxford paths still require historical datasets, checkpoints and in some branches missing modules (`boost_utils`, `loss.lpips`, `gaussian.encoder`/`decoder`). Older scripts contain machine-specific paths/GPU choices. Treat them as research sources requiring separate setup/review, not verified commands for the current environment. No KITTI weights or numerical claims are newly certified by this release.
