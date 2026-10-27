# KITTI: data preparation, training, and evaluation

KITTI uses two training settings: **GPS0** disables the GPS loss and **GPS1** enables it. Each trained model is evaluated on **test1 (Same-Area)** and **test2 (Cross-Area)**. Unlike VIGOR, a separate model is not trained for each test area.

The commands below describe the existing KITTI implementation. This documentation update checks its paths and command-line options; it does not claim a new KITTI training run or reproduction of the paper's numerical results.

## Prepare the data

Obtain the KITTI raw drives and satellite tiles following [Loc² datasets](https://github.com/vita-epfl/Loc2#datasets) and [HighlyAccurate](https://github.com/YujiaoShi/HighlyAccurate). Keep the repository's `dataLoader/train_files.txt`, `test1_files.txt`, and `test2_files.txt` unchanged.

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

Calibration files belong under the **date directory**, not inside each drive. The loader reads both original RGB (`data/`) and sky-masked RGB (`grd_no_sky/`), as well as metric depth. Keep matching frame names in each directory.

### Generate metric depth

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

The depth mask retains V1 predictions **>0.05**, after resizing to 256×1024, and multiplies the V2 metric depth by that mask. This differs from VIGOR's exact-zero mask rule. **This script does not write `grd_no_sky/` RGB images.** That directory must be prepared separately; its original generator is not bundled here. Do not assume the VIGOR mask command completes KITTI preprocessing or replace missing masked images with original RGB without reporting the change.

## Configure paths and dependencies

Run commands from the BevSplat repository root after the [shared installation](../README.md#install-the-environment). Build `feature_gaussian/` using `scripts/bootstrap_cuda.sh`.

- Dataset path: set `root_dir` in `dataLoader/KITTI_dataset.py`. The similarly named `KITTI_ROOT` constant in `kitti_main/config.py` is not passed to the actual loader.
- Output path: set `CKPT_ROOT` in `kitti_main/config.py` to your experiment directory.
- Rotation initialization: pass an explicit Stage-0 checkpoint with `--stage1_init`.

KITTI's Gaussian backbone loads DINO ViT-B/8 and ResNet50 through Torch Hub. It still requests `facebookresearch/dino:main` directly, so the VIGOR `BEVSPLAT_DINO_REPO` override does not cover every KITTI module. Allow these downloads on first use, or prepare the Torch Hub cache separately before offline execution. The VIGOR asset manifest alone is not a complete offline KITTI setup.

## Train the models

### 1. Prepare the rotation estimator (Stage 0)

Use an existing Stage-0 checkpoint, or the historical pretraining entry:

```bash
python train_KITTI_weak_nips.py --stage 0 \
  --rotation_range 10 --level 1 --channels 32_16_4 \
  --epochs 5 --name feat32
```

Before using this legacy script, edit its hardcoded `CUDA_VISIBLE_DEVICES='1'` and checkpoint output root in `getSavePath()` for your machine. It does not use the Stage-1 `--cuda` option. Set `--level 1` explicitly so its output matches the Stage-1 initialization layout; the legacy default is different. Use the resulting `model_4.pth` in the next step.

### 2. Train with and without GPS loss (Stage 1)

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

Each command trains one localization model, saves `model_0.pth` through `model_9.pth`, and evaluates both test splits after each epoch. Use a new name/output directory for each run. KITTI's seed and optimizer behavior are separate from the VIGOR settings; do not apply the VIGOR seed/LR recipe to these commands.

## Download weights and evaluate

The [OneDrive parent folder](https://1drv.ms/f/c/86d953bfc66eb903/IgAP7P2tFzChR7rHeMuXIOq8AakOxR02eKMyI2Z7qsMjLxo?e=zaD0Fb) contains the historical KITTI checkpoints below. They are outside the newer VIGOR release subfolder.

| Setting | File |
|---|---|
| GPS loss disabled | `KITTI_no_GPS.pth` |
| GPS loss enabled | `KITTI_GPS.pth` |

Download them into `checkpoints/` or use absolute checkpoint paths:

```bash
test -f checkpoints/KITTI_no_GPS.pth && python -m kitti_main.train --cuda 0 --test 1 \
  --rotation_range 0 --GPS_error_coe 0 \
  --ckpt checkpoints/KITTI_no_GPS.pth --name evaluate_kitti_gps0

test -f checkpoints/KITTI_GPS.pth && python -m kitti_main.train --cuda 0 --test 1 \
  --rotation_range 0 --GPS_error_coe 1 \
  --ckpt checkpoints/KITTI_GPS.pth --name evaluate_kitti_gps1
```

Each command evaluates test1 and test2 and writes their MAT predictions and result text files. **These examples use aligned headings**: `--rotation_range 0` disables heading perturbation/estimation and makes angle errors zero by construction. They are translation evaluations, not evidence of unknown-heading 3DoF performance. For a heading-perturbation experiment, select a nonzero range consistently for the data and model and report that range; do not compare the two protocols as if they were identical.

The KITTI loader is still permissive when loading checkpoints: a missing path can warn and continue, and incompatible/missing keys may be skipped. Confirm that the file exists and inspect loading counts/skips before trusting results. The stricter VIGOR checkpoint checks do not apply here. The two KITTI files' provider names/sizes were checked for this documentation update; their tensors and dataset metrics were not reevaluated.
