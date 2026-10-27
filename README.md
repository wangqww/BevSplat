# BevSplat

PyTorch implementation of [Resolving Height Ambiguity via Feature-Based Gaussian Primitives for Weakly-Supervised Cross-View Localization](https://arxiv.org/abs/2502.09080) (NeurIPS 2025).

This repository contains experiments on **KITTI** and **VIGOR**. The guides below cover data preparation, training, and evaluation.

| Dataset | Training settings | Evaluation | Detailed guide |
|---|---|---|---|
| KITTI | Two models: GPS loss disabled / enabled | Evaluate each model on Same-Area and Cross-Area | [KITTI guide](kitti_main/README.md) |
| VIGOR | Four models: Same/Cross-Area × GPS loss disabled/enabled | Train a separate model for each setting | [VIGOR data preparation](docs/data-preparation.md) |

**GPS0** means the GPS loss coefficient is 0; **GPS1** means it is 1. The current VIGOR models use the corrected implementation described below and are separate from the older paper checkpoints. All four current VIGOR models have completed training and evaluation; their final weights are available below.

## Install the environment

The current VIGOR experiments used Linux, Python **3.10.13**, PyTorch **2.4.1**, torchvision **0.19.1**, and CUDA **11.8**. A compatible CUDA toolkit/compiler is needed for the rasterizers.

```bash
git clone https://github.com/wangqww/BevSplat.git
cd BevSplat
conda create -n bevsplat-vigor python=3.10.13 -y
conda activate bevsplat-vigor
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements-vigor.txt
bash scripts/bootstrap_cuda.sh
python -c "import feat_gaussian, pano_gaussian_feat; print('extensions imported')"
```

Download the public pretrained models following [public dependencies](docs/public-dependencies.md), then set the paths below. These models are needed for both training and VIGOR evaluation. KITTI-specific paths and Torch Hub cache behavior are described in the [KITTI guide](kitti_main/README.md#configure-paths-and-dependencies).

```bash
export BEVSPLAT_FIT3D_REPO="$PWD/external/FiT3D"
export BEVSPLAT_DINO_REPO="$PWD/external/dino"
export TORCH_HOME="$PWD/.cache/torch"
export HF_HOME="$PWD/.cache/huggingface"
export HF_HUB_OFFLINE=1
export VIGOR_ROOT=/absolute/path/to/VIGOR
export VIGOR_CKPT_ROOT="$PWD/outputs"
python -m vigor_main.train --help
```

## Prepare the data

### KITTI

KITTI uses raw driving images, satellite tiles, calibration/GPS files, sky-masked RGB images, and metric-depth tensors. Follow the [KITTI data guide](kitti_main/README.md#prepare-the-data) for the directory layout, dataset sources, and Depth Anything preprocessing commands. The bundled split lists are `dataLoader/train_files.txt`, `test1_files.txt` (Same-Area), and `test2_files.txt` (Cross-Area).

KITTI preprocessing uses **Depth Anything V1 for the sky mask** and **Depth Anything V2 for metric depth**. After generating the masked depth, run `python scripts/prepare_kitti_rgb.py --kitti-root /path/to/KITTI` to create the required `grd_no_sky/` RGB images. The [step-by-step guide](kitti_main/README.md#2-generate-sky-masked-rgb-grd_no_sky) includes a preview command and explains the exact pixel rule. This new RGB-generation step is fully specified; it does not claim to recreate historical masked images byte for byte.

### VIGOR

Follow [the VIGOR data guide](docs/data-preparation.md) for the original VIGOR images, corrected labels, pinned Depth Anything **V1 ViT-L** sky masking, and pinned UniK3D metric depth. Dataset images are not redistributed.

```text
$VIGOR_ROOT/
├── splits__corrected/<city>/*.txt
└── <city>/
    ├── satellite/
    ├── panorama/
    ├── pano_mask_sky/
    ├── sky_mask/
    ├── UniK3D_same_metric/
    └── UniK3D_cross_metric/
```

| Protocol | Train | Validation | Official test | Train/validation cities | Test cities |
|---|---:|---:|---:|---|---|
| Same | 42087 | 10522 | 52605 | All four | All four |
| Cross | 41216 | 10304 | 53694 | NewYork, Seattle | SanFrancisco, Chicago |

Validation is the ordered final 20% of the corrected training list, separate from official test. Verify label identities with [configs/vigor-labels.sha256](configs/vigor-labels.sha256). Same/Cross use their corresponding metric-depth directories.

## KITTI experiments

KITTI has two training stages: first pretrain the rotation estimator for 5 epochs, then train a localization model for 10 epochs with GPS loss disabled or enabled. Each localization model is evaluated on both test areas, so KITTI needs **two models**, rather than four area-specific models.

The [KITTI experiment guide](kitti_main/README.md#train-the-models) includes Stage-0 preparation, both training commands, pretrained model downloads, and evaluation commands. It also explains the difference between aligned-heading evaluation (`--rotation_range 0`) and evaluation with heading perturbations. KITTI results have not been rerun as part of this documentation update.

## Train the VIGOR models

Train each VIGOR setting for **15 epochs** using seed **20261022**, batch size **8**, and one GPU. All four settings use the same learning rates: DPT **1e-4**, other Gaussian parameters **1e-5**, and pretrained ResNet **1e-6**. The main DINO feature extractor is frozen; the Gaussian encoder, its active BatchNorm statistics, and both DPT heads are trained. Both GPS settings use `GPS_error=5`. Training uses float32 without gradient accumulation.

Run the following in **Bash**. `COMMON` holds the shared options; the four commands choose the area and GPS setting. Run them separately with new output names. They can run sequentially on one GPU; change `--cuda` if needed.

```bash
export PYTHONHASHSEED=20261022
COMMON=(
  --lr 1e-4 --batch_size 8 --epochs 15 --seed 20261022
  --gaussian_lr_scale .1 --gaussian_pretrained_lr_scale .1
  --geometry_prior_coe .1 --freeze_gaussian_bn 0
  --model_behavior corrected --translation_origin crop
  --confidence_mode inherit --gaussian_offset_max 3
  --gaussian_scale_max .3 --gaussian_opacity_max 1
  --rotation_range 0 --GPS_error 5 --amount 1
  --resume 0 --init_ckpt "" --continue_weights ""
  --completed_steps 0 --pilot_steps 0 --max_batches 0
)
python -m vigor_main.train "${COMMON[@]}" --cuda 0 \
  --area same --GPS_error_coe 0 --test 0 --ckpt "" \
  --zero_init_gaussian_offsets 1 --name reproduce_same_gps0
python -m vigor_main.train "${COMMON[@]}" --cuda 0 \
  --area same --GPS_error_coe 1 --test 0 --ckpt "" \
  --zero_init_gaussian_offsets 1 --name reproduce_same_gps1
python -m vigor_main.train "${COMMON[@]}" --cuda 0 \
  --area cross --GPS_error_coe 0 --test 0 --ckpt "" \
  --zero_init_gaussian_offsets 1 --name reproduce_cross_gps0
python -m vigor_main.train "${COMMON[@]}" --cuda 0 \
  --area cross --GPS_error_coe 1 --test 0 --ckpt "" \
  --zero_init_gaussian_offsets 1 --name reproduce_cross_gps1
```

The loader uses 64 workers. Same-Area has 5261 optimizer updates per epoch (last batch: 7 samples), 78915 total; Cross-Area has 5152 (last batch: 8 samples), 77280 total. Each epoch runs validation and saves a checkpoint. Use the checkpoint after **epoch 15**, named **`model_14.pth`** because file indices start at zero. The test set is not used to select a checkpoint or stop training.

Each run starts from public DINO/FiT3D/ResNet pretrained weights with a new optimizer and learning-rate schedule, and trains continuously for 15 epochs. Gaussian offset/scale limits apply to each coordinate. A fixed seed helps comparison but does not guarantee identical results across hardware or library versions. Saved model weights do **not** restore the complete optimizer and random-number state; `resume`/`continue_weights` are not exact training-state recovery.

## Download and evaluate VIGOR models

Open the [OneDrive release folder](https://1drv.ms/f/c/86d953bfc66eb903/IgAP7P2tFzChR7rHeMuXIOq8AakOxR02eKMyI2Z7qsMjLxo?e=zaD0Fb), then `corrected-e2-fresh15-20261027`. Check the filename against [the model list](checkpoints/corrected-e2-manifest.json) and verify the download with [SHA256SUMS](checkpoints/SHA256SUMS). The parent folder also contains the older KITTI/paper checkpoints; use the matching dataset guide and commands.

| Model | AREA | GPS | Checkpoint filename |
|---|---|---:|---|
| Same/GPS0 | same | 0 | VIGOR_same_GPS0_corrected_E2_model14.pth |
| Same/GPS1 | same | 1 | VIGOR_same_GPS1_corrected_E2_model14.pth |
| Cross/GPS0 | cross | 0 | VIGOR_cross_GPS0_corrected_E2_model14.pth |
| Cross/GPS1 | cross | 1 | VIGOR_cross_GPS1_corrected_E2_model14.pth |

```bash
# After download, run in the directory containing these four files:
sha256sum -c /absolute/path/to/BevSplat/checkpoints/SHA256SUMS

# Return to BevSplat and reuse COMMON/environment above. Example: Same/GPS1.
AREA=same
GPS=1
CKPT=/absolute/path/to/VIGOR_same_GPS1_corrected_E2_model14.pth
for SPLIT in val test; do
  python -m vigor_main.train "${COMMON[@]}" --cuda 0 \
    --area "$AREA" --GPS_error_coe "$GPS" --test 1 \
    --eval_split "$SPLIT" --ckpt "$CKPT" \
    --zero_init_gaussian_offsets 0 --name "verify_${AREA}_gps${GPS}_${SPLIT}"
done
```

VIGOR evaluation rejects missing or incompatible model weights. Keep the evaluation options matched to the downloaded model: the weight file does not store settings such as `corrected`/`legacy` behavior, crop decoding, confidence mode, or Gaussian ranges.

Evaluation saves predictions and ground truth in MAT files. It reports mean/median position error in meters and the percentage of errors strictly below 1, 3, and 5 meters. The distance calculation is float32 `sqrt((pred_u-gt_u)^2 + (pred_v-gt_v)^2)`. Report validation and official-test results separately. Older checkpoints used different training histories; do not interpret comparisons with them as controlled single-factor experiments.

## Tests and reproducibility

```bash
CUDA_VISIBLE_DEVICES= python -B -m unittest discover -s tests -p 'test_vigor_*.py' -v
```

The current VIGOR path was checked with CPU regressions and a real model-initialization check; all 994 initial tensors matched the training reference. Some tests require a GPU or internal experiment controllers and are skipped by default. We have not repeated complete training on a clean machine for this documentation release. VIGOR uses 2DoF localization with a fixed 128×128 BEV. The [KITTI guide](kitti_main/README.md) documents its own settings and remaining setup requirements. Older sequence/Oxford/scratch experiments may need additional dependencies and path changes.

## Citation and licenses

```bibtex
@article{wang2025bevsplat,
  title={BevSplat: Resolving height ambiguity via feature-based Gaussian primitives for weakly-supervised cross-view localization},
  author={Wang, Qiwei and Wu, Shaoxun and Shi, Yujiao},
  journal={Advances in Neural Information Processing Systems},
  volume={38},
  pages={156668--156696},
  year={2025}
}
```

Preserve third-party notices. The CUDA rasterizers carry their own research/non-commercial terms: [feature_gaussian/LICENSE.md](feature_gaussian/LICENSE.md), [pano_feature_gaussian/LICENSE.md](pano_feature_gaussian/LICENSE.md). Public backbone code, weights and datasets retain their upstream licenses; no blanket MIT grant is made here.
