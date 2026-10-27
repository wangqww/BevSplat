# BevSplat

PyTorch implementation of [Resolving Height Ambiguity via Feature-Based Gaussian Primitives for Weakly-Supervised Cross-View Localization](https://arxiv.org/abs/2502.09080) (NeurIPS 2025).

This release documents the **corrected VIGOR E2 recipe**, including sky masking, metric-depth preparation, and four Same/Cross × GPS0/GPS1 experiments. These runs are separate from the paper's historical legacy checkpoints. The newest Same/GPS0 fixed endpoint is pending completion; intermediate weights are not released as final.

## Installation

The current experiments used Linux, Python **3.10.13**, PyTorch **2.4.1**, torchvision **0.19.1**, and CUDA **11.8**. A compatible CUDA toolkit/compiler is needed for the rasterizers.

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

Prepare the pinned public backbones and caches using [public dependencies](docs/public-dependencies.md). Evaluation also constructs these modules before loading a localization checkpoint.

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

## Data preparation

Follow [data preparation](docs/data-preparation.md) for the original VIGOR images, corrected labels, pinned Depth Anything **V1 ViT-L** sky masking, and pinned UniK3D metric depth. Dataset images are not redistributed.

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

## Four fresh15 training runs

All four use seed **20261022**, batch **8**, one GPU, no accumulation, native float32, and 15 continuous epochs. Peak learning rates: DPT **1e-4**, other Gaussian parameters **1e-5**, public-pretrained ResNet **1e-6**. The main DINO extractor stays frozen; the Gaussian encoder, its used BatchNorm statistics, and both DPT heads train. GPS0/1 is the GPS loss coefficient, with `GPS_error=5` in both.

Run these commands in **Bash**, each in a fresh process with a new output name. They may run sequentially on one GPU; change `--cuda` to an available device when needed.

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

The loader uses 64 workers. Same has 5261 updates/epoch (last batch 7), 78915 total; Cross has 5152 (last batch 8), 77280 total. Every epoch runs full validation and saves `model_<index>.pth`. The declared endpoint is **model_14.pth** (human epoch 15), without test-based model selection or early stopping.

Fresh means a new VIGOR optimizer/schedule and continuous RNG state, retaining public DINO/FiT3D/ResNet initialization. It does not mean all-random initialization. Offset/scale limits are per coordinate, not vector-norm limits. Fixed seeds do not guarantee bitwise equality across CUDA executions, hardware or library versions. Model-only checkpoints do **not** contain complete optimizer/RNG resume state; historical `resume`/`continue_weights` paths are not exact resume.

## Fixed-endpoint evaluation and weights

Open the [OneDrive release folder](https://1drv.ms/f/c/86d953bfc66eb903/IgAP7P2tFzChR7rHeMuXIOq8AakOxR02eKMyI2Z7qsMjLxo?e=zaD0Fb), then `corrected-e2-fresh15-20261027`. Match the filename and SHA256 in [the checkpoint manifest](checkpoints/corrected-e2-manifest.json) and [SHA256SUMS](checkpoints/SHA256SUMS). Legacy files in the parent folder belong to a different forward recipe.

| Model | AREA | GPS | Checkpoint filename |
|---|---|---:|---|
| Same/GPS0 | same | 0 | Pending verified final model14 |
| Same/GPS1 | same | 1 | VIGOR_same_GPS1_corrected_E2_model14.pth |
| Cross/GPS0 | cross | 0 | VIGOR_cross_GPS0_corrected_E2_model14.pth |
| Cross/GPS1 | cross | 1 | VIGOR_cross_GPS1_corrected_E2_model14.pth |

```bash
# After download, run in the directory containing these three files:
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

Public evaluation requires a complete matching state dictionary; missing/incompatible weights are errors. Runtime settings (corrected/legacy, crop decoding, confidence and Gaussian ranges) are not encoded in state dictionaries, so retain the matching command.

The evaluator saves prediction/GT MAT vectors. Native metrics use float32 `sqrt((pred_u-gt_u)^2 + (pred_v-gt_v)^2)` in meters: mean, median, and strict `<1`, `<3`, `<5` meter rates. Validation is not official test. Historical initialization and optimizer histories differ, so comparisons do not prove a single-factor causal effect or reproduce historical training bit for bit.

## Verification and supported scope

```bash
CUDA_VISIBLE_DEVICES= python -B -m unittest discover -s tests -p 'test_vigor_*.py' -v
```

CPU regressions and one real fresh CPU constructor were checked in the recorded environment; all 994 initial tensors matched the frozen training initial-state digest. GPU/private-controller tests are explicit opt-ins/skips. A clean-machine full training rerun was not performed for this packaging release. The supported current path is VIGOR 2DoF, fixed 128×128 BEV rasterization. Legacy KITTI/sequence/Oxford and scratch scripts are retained historical code, with unresolved dependencies/hardcoded paths outside this verified recipe; see [kitti_main](kitti_main/README.md).

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
