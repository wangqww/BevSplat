# KITTI: data preparation, configuration, and evaluation

[Repository setup](../README.md#install-the-environment) · [VIGOR guide](../vigor_main/README.md) · [File inventory](../docs/reproduction-files.md)

KITTI has two training settings: **GPS0** disables the GPS loss and **GPS1** enables it. Each trained model is evaluated on **test1 (Same-Area)** and **test2 (Cross-Area)**. For downloaded-weight evaluation, complete sections 1–4; Stage0/Stage1 training is optional.

This guide documents the actual existing code and a specified new RGB-generation recipe. It does not certify historical KITTI input equality or reproduced paper scores; see section 6.

## 1. Required files

Use the [KITTI inventory](../docs/reproduction-files.md#kitti-inputs) for the exact paths and role of every file. Both GPS settings require raw left-camera RGB, satellite tiles, date-level calibration, OXTS, masked RGB and metric depth, plus the fixed split lists. Download both historical localization checkpoints for all four model/split evaluations. Stage0 `model_4.pth` is needed only when training Stage1 with a nonzero heading range; it is not required for the evaluation-only path.

## 2. Prepare the data

Obtain the KITTI raw drives and satellite tiles following [Loc² datasets](https://github.com/vita-epfl/Loc2#datasets) and [HighlyAccurate](https://github.com/YujiaoShi/HighlyAccurate). Keep the repository's fixed split files unchanged. `train_files.txt` has 19655 rows, test1 has 3773 and test2 has 7542. Test rows include the prescribed shift/heading values; do not replace them with path-only lists.

```bash
# From the BevSplat root, verify exact bundled split bytes.
sha256sum -c configs/kitti-labels.sha256
```

```text
KITTI/
├── satmap/<date>/<drive>/<frame>.png
└── depth_data/<date>/
    ├── calib_cam_to_cam.txt
    └── <drive>/
        ├── image_02/data/<frame>.png
        ├── image_02/grd_no_sky/<frame>.png
        ├── image_02/grd_depth/<frame>_grd_depth.pt
        └── oxts/data/<frame>.txt
```

Calibration files belong under the **date directory**, not inside each drive. Keep `P_rect_02`; OXTS files supply GPS and heading even for GPS0. Satellite tiles must match the supplied dataset's original centers, directions and scale: `data_utils.py` assumes zoom 18, scale 2, 512 pixels and reference latitude 49.015, plus left-camera GPS offset `[1.08,0.26]` m. Arbitrary same-location satellite downloads are not equivalent. The loader reads both original RGB (`data/`) and sky-masked RGB (`grd_no_sky/`), as well as metric depth. Keep matching frame names in each directory.

### Generate the sky mask and metric depth

Use the pinned [Depth Anything companion](https://github.com/eacsai/Depth-Anything-V2) in a compatible preprocessing environment:

```bash
git clone https://github.com/eacsai/Depth-Anything-V2.git external/Depth-Anything-V2
git -C external/Depth-Anything-V2 checkout 25ed77cad6983520960bde74ace90e65c6a8db41
cd external/Depth-Anything-V2
pip install -r metric_depth/requirements.txt
mkdir -p metric_depth/checkpoints
# Download the two weights listed below, then update the three paths
# in metric_depth/predict_KITTI_depth.py before running it.
cd metric_depth
python predict_KITTI_depth.py
```

| Required file | Download |
|---|---|
| `depth_anything_v2_metric_vkitti_vitl.pth` | [Depth Anything V2 Metric VKITTI Large](https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-VKITTI-Large) |
| `depth_anything_vitl14.pth` | [Depth Anything V1 ViT-L](https://huggingface.co/LiheYoung/depth_anything_vitl14) |

Set `depth_v2_load_from`, the V1 `torch.load(...)` path, and `root_directory` in that script to your checkpoint files and `KITTI/depth_data`. It is a CUDA script, not a CPU entry point. It scans the configured KITTI dates and saves 256×1024 tensors to `image_02/grd_depth/<frame>_grd_depth.pt`; it also writes `grd_height/`, which the main KITTI localization loader does not require.

For reproducible checkpoint downloads, use the exact filenames and checksums in the [preprocessing asset inventory](../docs/reproduction-files.md#preprocessing-assets); the V1 file is downloaded as `pytorch_model.bin` upstream and renamed locally. The V2 hash there is a pinned public-provider identity, not proof of the historical training-preprocessing checkpoint bytes.

The depth mask retains raw V1 predictions **>0.05**, after bilinear resizing to 256×1024 with `align_corners=False`, and multiplies the V2 metric depth by that mask. This differs from VIGOR's exact-zero mask rule. V1 supplies the sky mask; V2 supplies metric depth. The script writes the masked depth tensors, not RGB images. Generate the RGB input in the next step.

### Generate sky-masked RGB (`grd_no_sky`)

From the BevSplat repository root, use the masked depth produced above to create the missing RGB images. The helper runs entirely on **CPU** and uses the shared PyTorch, Pillow, and NumPy dependencies; it does not run either depth model again.

```bash
export KITTI_ROOT=/absolute/path/to/KITTI

# Preview one sorted frame in a separate directory; dataset files are preserved.
python scripts/prepare_kitti_rgb.py --kitti-root "$KITTI_ROOT" \
  --output-root /absolute/path/to/kitti-rgb-preview --limit 1

# After inspecting the preview, generate all frames under the dataset root.
python scripts/prepare_kitti_rgb.py --kitti-root "$KITTI_ROOT"
```

For each frame, the helper reads `depth_data/<date>/<drive>/image_02/data/<frame>.png` and the corresponding `grd_depth/<frame>_grd_depth.pt`. It converts the original image to RGB, resizes it to **1024×256 with Pillow bilinear interpolation**, retains pixels with **masked metric depth >0**, and sets pixels with **depth ==0** to black. It writes a lossless RGB PNG to `image_02/grd_no_sky/<frame>.png`, with the same frame name and depth-aligned dimensions. With `--output-root`, the same `depth_data/...` layout is created under the preview root. Open the preview PNG and check the reported `processed`, `unchanged`, and `failed` counts; a failed frame causes a nonzero exit status.

Depth must be a finite, nonnegative float32 tensor of shape `(256, 1024)`. Raw RGB and depth files are never modified. An identical existing output is left unchanged; a different or invalid existing image is refused. Use a separate output root when comparing with historical `grd_no_sky` images rather than overwriting them.

**This is a new, explicitly defined RGB-generation recipe based on the stored masked depth.** It also blackens any other zero-depth pixels, so it is not guaranteed to equal the raw V1 sky mask alone. The previous README documented V1 sky masking and V2 depth generation, but the historical RGB-writing step was not recovered. Do not claim these newly generated PNGs are pixel-identical to historical KITTI inputs, or that existing KITTI checkpoints have been reevaluated on them. VIGOR uses its separate panorama-mask command.

## 3. Configure paths and dependencies

Run commands from the BevSplat repository root after the [shared installation](../README.md#install-the-environment) and [public asset setup](../docs/public-dependencies.md). Build `feature_gaussian/` using `scripts/bootstrap_cuda.sh`.

- Dataset path: set `root_dir` in `dataLoader/KITTI_dataset.py`. The similarly named `KITTI_ROOT` constant in `kitti_main/config.py` is not passed to the actual loader.
- `KITTI_ROOT` is only a shell convenience for preprocessing commands; the localization loader does not consume it. Set `root_dir` explicitly.
- Output path: set `CKPT_ROOT` in `kitti_main/config.py` to your experiment directory.
- Rotation initialization: pass an explicit Stage-0 checkpoint with `--stage1_init`.

KITTI's Gaussian backbone loads DINO ViT-B/8 and ResNet50 through Torch Hub. It still requests `facebookresearch/dino:main` directly, so the VIGOR `BEVSPLAT_DINO_REPO` override does not cover every KITTI module. Allow these downloads on first use, or prepare the Torch Hub cache separately before offline execution. The VIGOR asset manifest alone is not a complete offline KITTI setup.


```bash
export BEVSPLAT_FIT3D_REPO="$PWD/external/FiT3D"
export TORCH_HOME="$PWD/.cache/torch"
export HF_HOME="$PWD/.cache/huggingface"
# First-use KITTI Torch Hub downloads may need network access; see above.
python -m kitti_main.train --help
```

## 4. Download weights and evaluate

The [OneDrive parent folder](https://1drv.ms/f/c/86d953bfc66eb903/IgAP7P2tFzChR7rHeMuXIOq8AakOxR02eKMyI2Z7qsMjLxo?e=zaD0Fb) contains the historical KITTI checkpoints below. They are outside the newer VIGOR release subfolder.

| Setting | File |
|---|---|
| GPS loss disabled | `KITTI_no_GPS.pth` |
| GPS loss enabled | `KITTI_GPS.pth` |

Download them into `checkpoints/` or use absolute checkpoint paths:

```bash
set -euo pipefail
test -f checkpoints/KITTI_no_GPS.pth && python -m kitti_main.train --cuda 0 --test 1 \
  --rotation_range 0 --GPS_error_coe 0 \
  --ckpt checkpoints/KITTI_no_GPS.pth --name evaluate_kitti_gps0

test -f checkpoints/KITTI_GPS.pth && python -m kitti_main.train --cuda 0 --test 1 \
  --rotation_range 0 --GPS_error_coe 1 \
  --ckpt checkpoints/KITTI_GPS.pth --name evaluate_kitti_gps1
```

Each command evaluates test1 and test2 and writes their MAT predictions and result text files. **These examples use aligned headings**: `--rotation_range 0` disables heading perturbation/estimation and makes angle errors zero by construction. They are translation evaluations, not evidence of unknown-heading 3DoF performance. For a heading-perturbation experiment, select a nonzero range consistently for the data and model and report that range; do not compare the two protocols as if they were identical.

The KITTI loader is still permissive when loading checkpoints: a missing path can warn and continue, and incompatible/missing keys may be skipped. Confirm that the file exists and inspect loading counts/skips before trusting results. The stricter VIGOR checkpoint checks do not apply here. The two KITTI files' provider names/sizes were checked for this documentation update; their tensors and dataset metrics were not reevaluated.

## 5. Train the models

### Prepare the rotation estimator (Stage 0)

Use an existing Stage-0 checkpoint, or the historical pretraining entry:

```bash
python train_KITTI_weak_nips.py --stage 0 \
  --rotation_range 10 --level 1 --channels 32_16_4 \
  --epochs 5 --name feat32
```

Before using this legacy script, edit its hardcoded `CUDA_VISIBLE_DEVICES='1'` and checkpoint output root in `getSavePath()` for your machine. It does not use the Stage-1 `--cuda` option. Set `--level 1` explicitly so its output matches the Stage-1 initialization layout; the legacy default is different. Use the resulting `model_4.pth` in the next step. This file is the complete Stage-0 model state; Stage1 imports all compatible keys, including Gaussian/DPT initialization, not only the rotation head. The heading branch is frozen during Stage1. Stage0 still needs the same prepared RGB/depth inputs.

### Train with and without GPS loss (Stage 1)

```bash
STAGE0=/absolute/path/to/rotation-pretraining/model_4.pth
test -f "$STAGE0" && python -m kitti_main.train --cuda 0 \
  --GPS_error_coe 0 --GPS_error 5 --rotation_range 10 \
  --lr 6.25e-5 --batch_size 8 --epochs 10 \
  --stage1_init "$STAGE0" --name kitti_gps0

test -f "$STAGE0" && python -m kitti_main.train --cuda 0 \
  --GPS_error_coe 1 --GPS_error 5 --rotation_range 10 \
  --lr 6.25e-5 --batch_size 8 --epochs 10 \
  --stage1_init "$STAGE0" --name kitti_gps1
```

Each command trains one localization model, saves `model_0.pth` through `model_9.pth`, and evaluates both test splits after each epoch. Use a new name/output directory for each run. These are the existing code settings, not a newly validated paper reproduction. KITTI's seed and optimizer behavior are separate from the VIGOR settings; do not apply the VIGOR seed/LR recipe to these commands.

## 6. Results and verification

KITTI writes `test1_result.mat`, `test2_result.mat`, `test1_results.txt` and `test2_results.txt` under the printed run directory. MAT files contain `pred_lons`, `pred_lats`, `pred_oriens`, `gt_lons`, `gt_lats`, `gt_oriens`. TXT metrics include position/lateral/longitudinal errors in meters and angle errors in degrees, with strict `<1/3/5` thresholds. TXT logs append, so choose a new `--name` for each evaluation.

No verified numerical KITTI reference is attached to this release. In particular, the historical KITTI checkpoints have not been evaluated on the new depth-derived `grd_no_sky` recipe, and zero-range aligned-heading commands cannot establish perturbed-heading paper results. Do not publish the output as a reproduced paper row without matching the checkpoint, input corpus and heading protocol.

To claim exact dataset/result reproduction, still needed are the historical RGB-writing recipe or matching input identities, a frozen satellite/raw/preprocessed-input manifest, complete preprocessing environment/batch provenance, KITTI checkpoint byte identities and complete-loading validation, and a full clean-machine evaluation using that dataset. These are verification gaps, not instructions to alter images, labels, metrics or checkpoint loading to match a target.
