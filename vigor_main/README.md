# VIGOR: data preparation, configuration, and evaluation

[Repository setup](../README.md#install-the-environment) · [KITTI guide](../kitti_main/README.md) · [File inventory](../docs/reproduction-files.md)

This guide covers the four current 15-epoch VIGOR models: Same/Cross-Area × GPS0/GPS1. They use **corrected behavior and crop decoding**, and are separate from the historical paper checkpoints. For testing downloaded models, complete sections 1–4; localization training is optional.

## 1. Required files

See the [VIGOR file inventory](../docs/reproduction-files.md#vigor-inputs) and [public initialization assets](../docs/public-dependencies.md). You need original satellite/panorama images, **matching corrected lists**, sky-masked panorama JPEGs, protocol-specific metric NPYs, all model-construction assets and the matching final localization checkpoint. `sky_mask/` PNGs are diagnostic outputs, not loader inputs.

## 2. Prepare the data

Obtain original images through the [official VIGOR instructions](https://github.com/Jeff-Zilence/VIGOR/blob/main/data/DATASET.md). Obtain the corrected list files from [SliceMatch](https://github.com/tudelft-iv/SliceMatch/tree/73d27e95e4e808f8259a7e3a7bbf9944afbbc948/VIGOR_corrected_labels). The 16 files at commit `73d27e95e4e808f8259a7e3a7bbf9944afbbc948` have been individually downloaded and their SHA256 values match [configs/vigor-labels.sha256](../configs/vigor-labels.sha256). The repository supplies the hashes, not those corrected text files; the official original `splits.zip` must not be assumed to match them. Do not rename/reorder lists or randomly resplit.

```text
$VIGOR_ROOT/
├── splits__corrected/<city>/
│   ├── satellite_list.txt
│   ├── same_area_balanced_train__corrected.txt
│   ├── same_area_balanced_test__corrected.txt
│   └── pano_label_balanced__corrected.txt
└── <city>/
    ├── satellite/
    ├── panorama/
    ├── pano_mask_sky/
    ├── sky_mask/                  # optional diagnostic output
    ├── UniK3D_same_metric/
    └── UniK3D_cross_metric/
```

The four cities are NewYork, Seattle, SanFrancisco and Chicago. To obtain the corrected lists on a new dataset root, run the block below. If `splits__corrected` already exists, keep it and run the checksum check instead of copying over it. SliceMatch's model/training code is not used by BevSplat; only its corrected-label directory is needed. Please acknowledge the corrected-label authors as requested by their repository.

```bash
set -euo pipefail
export VIGOR_ROOT=/absolute/path/to/VIGOR
git clone https://github.com/tudelft-iv/SliceMatch.git external/SliceMatch
git -C external/SliceMatch checkout 73d27e95e4e808f8259a7e3a7bbf9944afbbc948
test ! -e "$VIGOR_ROOT/splits__corrected"
cp -R external/SliceMatch/VIGOR_corrected_labels/splits__corrected "$VIGOR_ROOT/"
```

Verify the 16 list files before preprocessing or evaluation:

```bash
# Run from BevSplat root; this check reads lists, not image/depth bytes.
export VIGOR_ROOT=/absolute/path/to/VIGOR
LABEL_SUMS="$PWD/configs/vigor-labels.sha256"
(cd "$VIGOR_ROOT" && sha256sum -c "$LABEL_SUMS")
```

Follow [VIGOR preprocessing](../docs/data-preparation.md), in order: pinned Depth Anything **V1 ViT-L** sky masking, then pinned **UniK3D** metric depth. UniK3D reads `pano_mask_sky/`, not original `panorama/`. Keep both original and masked RGB because the localization loader opens both. NPY files are float32 `(1,320,640)` named `<panorama-stem>_depth.npy`. Use the corresponding Same/Cross directory; do not substitute unrelated metric-depth exports.

| Protocol | Train | Validation | Official test | Train/validation cities | Test cities |
|---|---:|---:|---:|---|---|
| Same | 42087 | 10522 | 52605 | All four | All four |
| Cross | 41216 | 10304 | 53694 | NewYork, Seattle | SanFrancisco, Chicago |

Validation is the ordered final 20% of the source training list. The ordered first 80% trains the model. `--amount 1` retains the complete protocol. Do not compare an internal validation score with the official test score.

## 3. Configure paths and dependencies

Run from the repository root after [shared installation](../README.md#install-the-environment) and [public asset setup](../docs/public-dependencies.md). Set these in each new shell:

```bash
export BEVSPLAT_FIT3D_REPO="$PWD/external/FiT3D"
export BEVSPLAT_DINO_REPO="$PWD/external/dino"
export TORCH_HOME="$PWD/.cache/torch"
export HF_HOME="$PWD/.cache/huggingface"
export HF_HUB_OFFLINE=1
export VIGOR_ROOT=/absolute/path/to/VIGOR
export VIGOR_CKPT_ROOT="$PWD/outputs/vigor"
export PYTHONHASHSEED=20261022
python -m vigor_main.train --help
```

`dataLoader/Vigor_dataset_gs.py` reads `VIGOR_ROOT`; `vigor_main/config.py` reads `VIGOR_CKPT_ROOT`. Current commands explicitly override compatibility defaults. In particular, the CLI defaults to `legacy/legacy`, so always pass `--model_behavior corrected --translation_origin crop` for the four files below. A checkpoint does not encode these switches, Gaussian ranges or confidence mode.

## 4. Download weights and evaluate

Open the [OneDrive release folder](https://1drv.ms/f/c/86d953bfc66eb903/IgAP7P2tFzChR7rHeMuXIOq8AakOxR02eKMyI2Z7qsMjLxo?e=zaD0Fb), then `corrected-e2-fresh15-20261027`. Download these four files into one directory. Match [the manifest](../checkpoints/corrected-e2-manifest.json) and [SHA256SUMS](../checkpoints/SHA256SUMS); the parent folder's historical weights use different recipes.

| Model | AREA | GPS | Filename |
|---|---|---:|---|
| Same/GPS0 | same | 0 | VIGOR_same_GPS0_corrected_E2_model14.pth |
| Same/GPS1 | same | 1 | VIGOR_same_GPS1_corrected_E2_model14.pth |
| Cross/GPS0 | cross | 0 | VIGOR_cross_GPS0_corrected_E2_model14.pth |
| Cross/GPS1 | cross | 1 | VIGOR_cross_GPS1_corrected_E2_model14.pth |

This **Bash block is independent of the training section**. It defines all evaluation options and tests all four checkpoints; `val` is included separately for comparison with the internal validation results. Keep the environment in section 3 active.

```bash
set -euo pipefail
CHECKPOINT_DIR=/absolute/path/to/downloaded/checkpoints
CHECKPOINT_SUMS="$PWD/checkpoints/SHA256SUMS"
(cd "$CHECKPOINT_DIR" && sha256sum -c "$CHECKPOINT_SUMS")
EVAL=(
  --lr 1e-4 --batch_size 8 --epochs 15 --seed 20261022
  --gaussian_lr_scale .1 --gaussian_pretrained_lr_scale .1
  --geometry_prior_coe .1 --freeze_gaussian_bn 0
  --model_behavior corrected --translation_origin crop
  --confidence_mode inherit --gaussian_offset_max 3
  --gaussian_scale_max .3 --gaussian_opacity_max 1
  --rotation_range 0 --GPS_error 5 --amount 1
  --resume 0 --init_ckpt "" --continue_weights ""
  --completed_steps 0 --pilot_steps 0 --max_batches 0
  --test 1 --zero_init_gaussian_offsets 0
)
for AREA in same cross; do
  for GPS in 0 1; do
    CKPT="$CHECKPOINT_DIR/VIGOR_${AREA}_GPS${GPS}_corrected_E2_model14.pth"
    test -f "$CKPT"
    for SPLIT in val test; do
      python -m vigor_main.train "${EVAL[@]}" --cuda 0         --area "$AREA" --GPS_error_coe "$GPS" --eval_split "$SPLIT"         --ckpt "$CKPT" --name "verify_${AREA}_gps${GPS}_${SPLIT}"
    done
  done
done
```

Evaluation rejects incomplete/incompatible checkpoints. It writes `<area>_val_result.mat` or `<area>_result.mat` and corresponding result TXT files under the run directory printed at startup. The MAT vectors are `pred_us`, `pred_vs`, `gt_us`, `gt_vs`; they retain evaluation order. Mean/median error use float32 `sqrt((pred_u-gt_u)^2+(pred_v-gt_v)^2)` in meters; percentages use strict **<1/<3/<5 m**. Use a new `--name` for each evaluation so previous result files are not confused with the current run.

## 5. Train the models

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

## 6. Results and verification

The following are **official-test measurements on the author's stored input corpus**, using the exact final files above, corrected/crop behavior, zero rotation range and the complete official split. They are not newly regenerated-data or paper-table measurements.

| Model | N | Mean m | Median m | <1 m % | <3 m % | <5 m % |
|---|---:|---:|---:|---:|---:|---:|
| Same/GPS0 | 52605 | 3.1164 | 1.4841 | 31.2423 | 75.8426 | 86.0565 |
| Same/GPS1 | 52605 | 2.6241 | 1.2852 | 38.0705 | 79.8973 | 89.0980 |
| Cross/GPS0 | 53694 | 3.0487 | 1.4183 | 31.8937 | 81.9552 | 87.8813 |
| Cross/GPS1 | 53694 | 2.6637 | 1.2178 | 39.6823 | 84.8288 | 89.9654 |

All four final model identities and their independent val/test vectors were verified on the stored corpus. This does **not** establish that a new machine regenerates the same corpus or retrains identical weights. The current preprocessing bridge has a specified recipe, but its historical full-corpus batching, Torch RNG state, complete preprocessing dependency lock and per-input byte identities are not published. Train-mode UniK3D inference and JPEG encoding can change regenerated inputs. List hashes and checkpoint hashes alone cannot certify dataset identity. These gaps must be closed before promising the exact scores above on regenerated data. Do not silently change model mode or filtering to chase these numbers.
