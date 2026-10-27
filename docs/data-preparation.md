# VIGOR data preparation

Obtain the original images and corrected split lists through [VIGOR](https://github.com/Jeff-Zilence/VIGOR/blob/main/data/DATASET.md) and the [Loc2 data instructions](https://github.com/vita-epfl/Loc2#datasets). Preserve filenames and list order under `splits__corrected`; do not randomly resplit. From the dataset root, run `sha256sum -c /absolute/path/to/BevSplat/configs/vigor-labels.sha256`.

All four cities are NewYork, Seattle, SanFrancisco and Chicago. The training loader requires original `panorama`, masked `pano_mask_sky`, satellite images, and protocol-specific metric NPY files. Missing images now fail with the offending path instead of becoming black placeholders.

## Sky masking

Use the pinned companion repository in its compatible environment:

```bash
git clone https://github.com/eacsai/Depth-Anything-V2.git external/Depth-Anything-V2
git -C external/Depth-Anything-V2 checkout 25ed77cad6983520960bde74ace90e65c6a8db41
cd external/Depth-Anything-V2
pip install -r requirements.txt
mkdir -p metric_depth/checkpoints
curl -fL https://huggingface.co/LiheYoung/depth_anything_vitl14/resolve/main/pytorch_model.bin \
  -o metric_depth/checkpoints/depth_anything_vitl14.pth
echo "6c6a383e33e51c5fdfbf31e7ebcda943973a9e6a1cbef1564afe58d7f2e8fe63  metric_depth/checkpoints/depth_anything_vitl14.pth" | sha256sum -c -
python metric_depth/mask_vigor_sky.py --vigor-root "$VIGOR_ROOT"
cd ../..
```

Despite the repository name, this script loads **Depth Anything V1 ViT-L**. RGB is divided by 255, ImageNet-normalized and lower-bound resized to 518/divisible by 14. The raw prediction is bilinearly resized back (`align_corners=False`); `prediction != 0` retains RGB, exact zero becomes black. There is no min-max normalization, epsilon threshold or semantic segmentation model. Outputs retain original filenames/resolution. Auxiliary PNG masks use 255=sky and 0=keep; BevSplat consumes the masked JPEG, not that PNG. JPEG re-encoding can leave small nonzero boundary values; no bitwise historical-image equivalence is claimed. See the companion's CPU mask tests.

## Metric depth

Use a separate environment compatible with the pinned UniK3D dependency and its installation instructions; do not replace the localization environment's dependencies to run preprocessing.

```bash
git clone https://github.com/eacsai/UniK3Dnew.git external/UniK3Dnew
git -C external/UniK3Dnew checkout 83f63a31332e6998dabb2619a7c5b8358d17712a
# In the preprocessing environment, from external/UniK3Dnew:
pip install -e .
# Return to the BevSplat root before the commands below.
export HF_HOME="$PWD/.cache/unik3d-huggingface"
unset HF_HUB_OFFLINE
python - <<'PYMODEL'
import os, pathlib
from huggingface_hub import snapshot_download
revision = '008aa54308b8bc142f74c9c7b098833aaede06bd'
folder = pathlib.Path(snapshot_download('lpiccinelli/unik3d-vitl', revision=revision,
    cache_dir=os.environ['HF_HOME'] + '/hub'))
ref = folder.parent.parent / 'refs/main'
ref.parent.mkdir(parents=True, exist_ok=True)
ref.write_text(revision)
PYMODEL
export HF_HUB_OFFLINE=1
python scripts/prepare_vigor_depth.py --help
python scripts/prepare_vigor_depth.py --unik3d-root external/UniK3Dnew \
  --vigor-root "$VIGOR_ROOT" --area same --split train --device cpu \
  --batch-size 2 --limit 2 --output-root /absolute/new/preview --dry-run
```

The dry run reads lists/checks paths without loading a model or creating outputs. Remove `--dry-run` to generate a two-item preview. For complete generation, omit `--limit` and `--output-root`:

```bash
for AREA in same cross; do
  for SPLIT in train test; do
    python scripts/prepare_vigor_depth.py --unik3d-root external/UniK3Dnew \
      --vigor-root "$VIGOR_ROOT" --area "$AREA" --split "$SPLIT" \
      --device cuda:0 --batch-size 8
  done
done
```

`train` here covers the entire source list later divided into localization train/validation. Existing outputs cause an error unless `--overwrite` is explicit. The new bridge avoids the old dependency CLI's hardcoded CUDA2/root and `__getitem__` output side effects. Do not use its `vigor_plus`/Kansas entry as VIGOR.

The wrapper uses public `lpiccinelli/unik3d-vitl` (recorded revision `008aa54308b8bc142f74c9c7b098833aaede06bd`), Spherical 512×1024 camera, resolution level 9, raw RGB PIL bilinear resize to 640×320, normalization and distance output. It intentionally retains the original runner's **train mode** during no-grad inference. Negative depths, resized uint8 RGB sums below 16 and depths **>80 m** become zero (80 retained). Files are `(1,320,640)` at `<city>/UniK3D_<area>_metric/<panorama-stem>_depth.npy`.

The pinned companion includes device/path and batch-shape repairs with a CPU batch-2 smoke. This release checked the real ordered-path dry run; it did not regenerate the full corpus or certify historical NPY pixel equality. Train-mode inference, batching and dependency versions can affect regenerated values. Use the same generated corpus across compared localization runs and record its provenance.
