# BevSplat

**Resolving Height Ambiguity via Feature-Based Gaussian Primitives for Weakly-Supervised Cross-View Localization**

[![NeurIPS 2025 Spotlight](https://img.shields.io/badge/NeurIPS-2025_Spotlight-red.svg)](https://neurips.cc/virtual/2025/loc/san-diego/poster/118781)
[![arXiv](https://img.shields.io/badge/arXiv-2502.09080-b31b1b.svg)](https://arxiv.org/abs/2502.09080)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official PyTorch implementation of **BevSplat** (NeurIPS 2025 Spotlight).

> **TL;DR.** Each ground-image pixel is lifted into the world as `Np = 3` 3D Gaussian primitives whose feature/confidence channels come from a DINOv2 + DPT head; these Gaussians are rasterized orthographically into a Bird's-Eye-View (BEV) feature map and cross-correlated with the satellite tile to localize the camera. Training uses an InfoNCE-style softplus on correlation peaks plus an optional GPS-noise term — `L = L_Weakly + λ₁ · L_GPS` (paper Eq. 1).

---

## Contents

- [Quick start](#quick-start)
- [Datasets](#datasets)
- [Reproduce Table 1 (KITTI)](#reproduce-table-1-kitti)
- [Reproduce Table 2 (VIGOR)](#reproduce-table-2-vigor)
- [Architecture in 200 words](#architecture-in-200-words)
- [Repo layout](#repo-layout)
- [Notes on the legacy code](#notes-on-the-legacy-code)
- [Citation](#citation)

---

## Quick start

The two reproduction packages — `kitti_main/` and `vigor_main/` — are thin orchestration layers over the algorithm modules. Everything below assumes you have a CUDA-capable GPU (≥24 GB recommended for batch 8; 8 GB works at batch 1).

```bash
# 1. Install uv (once)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Clone the repo
git clone https://github.com/<your-org>/BevSplat.git
cd BevSplat

# 3. Create the Python env (Python 3.11, deps pinned to versions that produced the paper numbers)
uv venv
source .venv/bin/activate
uv sync                                       # ~2 min

# 4. Vendor glm headers and build the two CUDA rasterizers
bash scripts/bootstrap_cuda.sh                # ~5 min

# 5. Smoke-test
python -c "import feat_gaussian, pano_gaussian_feat; print('CUDA exts OK')"
python -m kitti_main.train --help             # should print usage
python -m vigor_main.train --help
```

The version pins in `pyproject.toml` target CUDA 11.8. If your toolkit is different, change the `pytorch-cu118` index URL (e.g. to `cu121`) and re-run `uv lock && uv sync`.

### Optional: skip uv

If you already have a working `torch>=2.6` + CUDA 11.8 environment you can install the Python deps with plain pip:

```bash
pip install torch==2.6.0 torchvision==0.21.0 timm==1.0.27 e3nn==0.6.0 \
            einops==0.8.1 jaxtyping==0.3.2 numpy==1.26.4 scipy==1.11.4 \
            pandas==2.1.4 scikit-image opencv-python==4.10.0.84 \
            open3d==0.19.0 plotly Pillow matplotlib
bash scripts/bootstrap_cuda.sh
```

### Have Claude Code (or any coding agent) do the setup

The repo ships a `CLAUDE.md` at the root that already documents the install layout, the empty-glm-submodule pitfall, and the two python deps (`e3nn`, `timm`) the upstream `requirements.txt` forgot. If you point [Claude Code](https://claude.com/claude-code) — or any coding agent with shell access — at this repo, the following prompt produces a working env in one shot. Adapt the path / CUDA notes for your machine.

> Set up the BevSplat repo so I can run `python -m kitti_main.train --help` and `python -m vigor_main.train --help` without errors. The repo is at `<absolute-path-to-clone>`; my CUDA toolkit is `<11.8 | 12.1 | 12.4>`; my GPU is `<e.g. RTX 4090, 24 GB>`.
>
> Read `README.md` and `CLAUDE.md` first. Then:
>
> 1. Use `uv` to create the venv (`uv venv`) and install the pinned Python deps (`uv sync`). If my CUDA toolkit is not 11.8, edit the `pytorch-cu118` index URL in `pyproject.toml` to match (`cu121` / `cu124`) before syncing.
> 2. Run `bash scripts/bootstrap_cuda.sh` to vendor the `glm` headers and build the two CUDA rasterizers in editable mode.
> 3. Smoke-test: `python -c "import feat_gaussian, pano_gaussian_feat, torch; print(torch.cuda.is_available())"`, then `python -m kitti_main.train --help` and `python -m vigor_main.train --help`.
> 4. If any step fails, surface the exact error and propose a fix — do **not** edit `gaussian/*.py`, `models/*.py`, `backbone/*.py`, `dataLoader/*.py`, `vis_gaussian_*.py`, `feature_gaussian/cuda_rasterizer/*`, or `pano_feature_gaussian/cuda_rasterizer/*`. Those are the algorithm; touching them invalidates the numbers in the README's reproduction tables.
> 5. Report back the exact versions installed (`uv pip freeze | grep -iE 'torch|timm|e3nn|open3d'`) and the output of the three smoke-test commands.

Things that commonly go wrong on a fresh machine and that the agent should already be aware of from `CLAUDE.md`:

- The `third_party/glm/` submodule is **not initialized** in the upstream commit; `bootstrap_cuda.sh` is what fixes that by vendoring glm 0.9.9.8.
- `gaussian/encoder_pano.py` top-level-imports `open3d` and `plotly` even though the model never calls them — they're in `pyproject.toml` for a reason; don't strip them.
- The hardcoded dataset roots (`KITTI_ROOT` / `VIGOR_ROOT`) live in `kitti_main/config.py` and `vigor_main/config.py`; agents should edit those files, not the legacy `dataLoader/*.py` ones, when pointing at a different filesystem layout.
- If `nvcc` isn't on `PATH`, the bootstrap script will fail with `FATAL: nvcc not on PATH`; agents should suggest `module load cuda/11.8` or equivalent rather than trying to work around it.

After it finishes, ask the agent to also run one of the eval-only commands from the [Reproduce Table 1](#reproduce-table-1-kitti) section against a checkpoint you point at — that's the fastest end-to-end correctness check (~10 minutes per split on a free 4090).

---

## Datasets

Both datasets are external; we don't redistribute them.

### KITTI

Expected layout under `${KITTI_ROOT}` (default: `/data/dataset/KITTI`):

```
${KITTI_ROOT}/
├── satmap/                                  # satellite tiles, indexed by drive/frame
└── depth_data/                              # KITTI raw drives + pre-computed depth
    └── <date>/<drive>/
        ├── image_02/data/                   # raw left RGB
        ├── image_02/grd_no_sky/             # sky-masked RGB (what the model consumes)
        ├── image_02/grd_depth/*_grd_depth.pt
        ├── oxts/data/                       # GPS/IMU oxts files
        └── calib_cam_to_cam.txt
```

Drive/file lists live in `dataLoader/{train_files,test1_files,test2_files}.txt` (already in the repo). `test1_files.txt` is the **Same-Area** split; `test2_files.txt` is **Cross-Area**.

Change the dataset root by editing `KITTI_ROOT` in `kitti_main/config.py` or `root_dir` in `dataLoader/KITTI_dataset.py:21`.

**Where the source data comes from.** We use the same KITTI assembly as [Loc²](https://github.com/vita-epfl/Loc2). Follow [Loc² → Datasets](https://github.com/vita-epfl/Loc2#datasets) (which in turn points at [HighlyAccurate](https://github.com/YujiaoShi/HighlyAccurate)) to organize the raw KITTI drives, the satellite tiles, and the train/test1/test2 file lists into `satmap/` + `depth_data/` as shown above.

**Generating per-frame depth.** Loc² doesn't ship per-pixel ground-image depth. The `image_02/grd_depth/*_grd_depth.pt` tensors expected by BevSplat are produced by running [Depth-Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) over every `image_02/grd_no_sky/*.png` in your KITTI tree, then saving each output to the matching `image_02/grd_depth/*_grd_depth.pt` location. (Approximately one `.pt` per RGB frame; total ~6 GB across the KITTI train + test1 + test2 splits.)

### VIGOR

Expected layout under `${VIGOR_ROOT}` (default: `/data/dataset/VIGOR`):

```
${VIGOR_ROOT}/
├── <city>/                                  # NewYork / Chicago / SanFrancisco / Seattle
│   ├── satellite/
│   ├── panorama/                            # original panoramas (unused by Stage 1)
│   ├── pano_mask_sky/                       # sky-masked panoramas (what the model consumes)
│   └── UniK3D_{same,cross}_metric/          # pre-computed metric depth as <id>_depth.npy
└── splits__corrected/
    └── <city>/
        ├── satellite_list.txt
        ├── same_area_balanced_{train,test}__corrected.txt
        └── pano_label_balanced__corrected.txt
```

Change the dataset root in `vigor_main/config.py` or `dataLoader/Vigor_dataset_gs.py:15`.

**Where the source data comes from.** We use the same VIGOR assembly as [Loc²](https://github.com/vita-epfl/Loc2). Follow [Loc² → Datasets](https://github.com/vita-epfl/Loc2#datasets) (which points at the [official VIGOR repo](https://github.com/Jeff-Zilence/VIGOR/blob/main/data/DATASET.md)) to obtain `satellite/`, `panorama/`, `pano_mask_sky/`, and `splits__corrected/`.

**Generating per-frame depth.** The `UniK3D_{same,cross}_metric/<id>_depth.npy` tensors are produced by [UniK3D](https://github.com/lpiccinelli-eth/UniK3D); we ship the VIGOR-specific glue (dataset wrapper, model wrapper, runner) in a companion repo at **[eacsai/UniK3Dnew](https://github.com/eacsai/UniK3Dnew)**. Clone it next to BevSplat and run:

```bash
git clone https://github.com/eacsai/UniK3Dnew.git
cd UniK3Dnew
pip install -e .                              # installs unik3d + its torch deps

# Edit dataset_vigor.py:18 — set `root` to your VIGOR_ROOT.
# Then generate depth for each (area, split) combination you need:
python depth_vigor.py --area same  --train 0 --batch_size 4   # same-area test split
python depth_vigor.py --area same  --train 1 --batch_size 4   # same-area train split
python depth_vigor.py --area cross --train 0 --batch_size 4   # cross-area test split
python depth_vigor.py --area cross --train 1 --batch_size 4   # cross-area train split
```

Each invocation writes `${VIGOR_ROOT}/<city>/UniK3D_{same|cross}_metric/<id>_depth.npy` for every panorama in the chosen split — which is exactly what BevSplat's `dataLoader/Vigor_dataset_gs.py` expects, no renaming required.

---

## Pre-trained checkpoints

We ship the six checkpoints that produced the numbers in this README's reproduction tables on OneDrive: **[Wangqw's Neurips2025 (BevSplat)](https://1drv.ms/f/c/86d953bfc66eb903/IgAP7P2tFzChR7rHeMuXIOq8AakOxR02eKMyI2Z7qsMjLxo?e=zaD0Fb)**.

| File | Size | Reproduces |
|---|---|---|
| `KITTI_no_GPS.pth`           | 1.11 GB | KITTI Table 1, λ₁ = 0 — used for both **Same-Area** (`test1`) and **Cross-Area** (`test2`) |
| `KITTI_GPS.pth`              | 1.11 GB | KITTI Table 1, λ₁ = 1 — used for both Same-Area and Cross-Area |
| `VIGOR_same_no_GPS.pth`      | 924 MB  | VIGOR Table 2, Same-Area Aligned-orientation, λ₁ = 0 |
| `VIGOR_same_GPS.pth.pth`     | 867 MB  | VIGOR Table 2, Same-Area Aligned-orientation, λ₁ = 1 |
| `VIGOR_cross_no_GPS.pth.pth` | 856 MB  | VIGOR Table 2, Cross-Area Aligned-orientation, λ₁ = 0 |
| `VIGOR_cross_GPS.pth.pth`    | 848 MB  | VIGOR Table 2, Cross-Area Aligned-orientation, λ₁ = 1 |

> The double `.pth.pth` suffix on four of the VIGOR files matches what was uploaded — keep the names as-is when you `torch.load(...)` them.

### Where to put the downloaded files

Drop everything into a `checkpoints/` directory at the repo root. The eval examples in the next sections expect them there:

```bash
mkdir -p checkpoints
# Open the OneDrive folder above and download all six .pth files into ./checkpoints/.
# OneDrive does not offer stable direct-download URLs from share links, so
# either download manually via the web UI or use a tool like `rclone` / `onedrive-cli`
# pointing at the shared folder.
ls -lh checkpoints/
```

If you'd rather store them elsewhere, just pass an absolute `--ckpt` path to the eval commands below.

---

## Reproduce Table 1 (KITTI)

Each row of the "Ours" rows in Table 1 maps to one Stage-1 checkpoint; **Same-Area** and **Cross-Area** share the same model and only differ in which test split is evaluated. The KITTI pipeline uses a 3-DoF protocol (lat / lon / heading), with a frozen Stage-0 rotation pre-head feeding heading estimates at train time.

### One-time setup — Stage-0 rotation pre-head

The Stage-1 trainer needs a Stage-0 checkpoint at `kitti_main.config.STAGE0_INIT_CKPT` (a 5-epoch self-supervised rotation regressor):

```bash
python train_KITTI_weak_nips.py --stage 0 --rotation_range 10 --epochs 5 --name feat32
```

This writes `Stage0/lat20.0m_lon20.0m_rot10.0_..._feat32/model_4.pth`. If you already have this `.pth`, skip the command and just point `--stage1_init` at it.

### Training (~10 h / row on a single 4090)

```bash
# Row "Ours" λ₁ = 0  (paper Same 5.82 / 2.85, Cross 7.05 / 3.22)
python -m kitti_main.train --GPS_error_coe 0 --rotation_range 10 \
        --epochs 10 --name reproduce_lambda0

# Row "Ours" λ₁ = 1  (paper Same 2.87 / 2.06, Cross 6.20 / 2.51)
python -m kitti_main.train --GPS_error_coe 1 --rotation_range 10 \
        --epochs 10 --name reproduce_lambda1
```

Each command trains one model under `${CKPT_ROOT}/Stage1/...` and evaluates `test1` (Same-Area) and `test2` (Cross-Area) after every epoch.

### Evaluation only (loads a saved `.pth`)

Drop the OneDrive `.pth` files into `checkpoints/` (see [Pre-trained checkpoints](#pre-trained-checkpoints)), then:

```bash
# λ₁ = 0 — evaluates test1 (Same-Area) and test2 (Cross-Area) in one run
python -m kitti_main.train --test 1 --rotation_range 0 --GPS_error_coe 0 \
        --ckpt checkpoints/KITTI_no_GPS.pth \
        --name verify_lambda0

# λ₁ = 1
python -m kitti_main.train --test 1 --rotation_range 0 --GPS_error_coe 1 \
        --ckpt checkpoints/KITTI_GPS.pth \
        --name verify_lambda1
```

### Numbers reproduced on the dev server

Each row below reproduces the matching paper cell within ≤0.5 m mean error using checkpoints trained per the recipe above.

| Setting | Mean (m) ↓ | Median (m) ↓ | Lat d=1 m % ↑ | Lat d=3 m % ↑ | Lon d=1 m % ↑ | Lon d=3 m % ↑ |
|---|---|---|---|---|---|---|
| Same λ=0 — paper      | 5.82 | 2.85 | 60.04 | 91.54 | 24.06 | 56.82 |
| Same λ=0 — repro      | 5.678 | 2.827 | 58.68 | 92.74 | 25.39 | 57.33 |
| Same λ=1 — paper      | 2.87 | 2.06 | 52.90 | 94.24 | 35.62 | 76.57 |
| Same λ=1 — repro      | 2.800 | 1.928 | 62.18 | 96.16 | 37.58 | 80.23 |
| Cross λ=0 — paper     | 7.05 | 3.22 | 58.15 | 92.62 | 23.08 | 51.61 |
| Cross λ=0 — repro     | 6.719 | 2.975 | 57.89 | 93.86 | 23.85 | 54.28 |
| Cross λ=1 — paper     | 6.20 | 2.51 | 51.45 | 95.17 | 27.41 | 60.45 |
| Cross λ=1 — repro     | 5.856 | 2.334 | 61.27 | 94.71 | 30.77 | 65.00 |

---

## Reproduce Table 2 (VIGOR)

VIGOR is 2-DoF (translation only); the GT heading is consumed directly by the BEV renderer. The "Aligned-orientation" columns are what `vigor_main` produces by default (`--rotation_range 0`); the "Unknown-orientation" columns require training with rotation perturbation and are not covered by the bundled recipes.

### Training (~10 h / row on a single 4090)

```bash
# Same-Area λ₁ = 0  (paper 3.15 / 1.45)
python -m vigor_main.train --area same  --GPS_error_coe 0 --lr 1.25e-4 --epochs 15 --name reproduce_lambda0

# Same-Area λ₁ = 1  (paper 2.87 / 1.58)
python -m vigor_main.train --area same  --GPS_error_coe 1 --lr 1e-4    --epochs 15 --name reproduce_lambda1

# Cross-Area λ₁ = 0 (paper 3.03 / 1.41)
python -m vigor_main.train --area cross --GPS_error_coe 0 --lr 6.5e-5  --epochs 15 --name reproduce_lambda0

# Cross-Area λ₁ = 1 (paper 2.84 / 1.36)
python -m vigor_main.train --area cross --GPS_error_coe 1 --lr 1e-4    --epochs 15 --name reproduce_lambda1
```

### Evaluation only

Drop the OneDrive `.pth` files into `checkpoints/`, then use the matching file + lr per row:

```bash
# Same-Area, λ₁ = 0
python -m vigor_main.train --test 1 --area same  --batch_size 2 --lr 1.25e-4 --GPS_error_coe 0 \
        --ckpt checkpoints/VIGOR_same_no_GPS.pth      --name verify_same_lambda0

# Same-Area, λ₁ = 1
python -m vigor_main.train --test 1 --area same  --batch_size 2 --lr 1e-4    --GPS_error_coe 1 \
        --ckpt checkpoints/VIGOR_same_GPS.pth.pth     --name verify_same_lambda1

# Cross-Area, λ₁ = 0
python -m vigor_main.train --test 1 --area cross --batch_size 2 --lr 6.5e-5  --GPS_error_coe 0 \
        --ckpt checkpoints/VIGOR_cross_no_GPS.pth.pth --name verify_cross_lambda0

# Cross-Area, λ₁ = 1
python -m vigor_main.train --test 1 --area cross --batch_size 2 --lr 1e-4    --GPS_error_coe 1 \
        --ckpt checkpoints/VIGOR_cross_GPS.pth.pth    --name verify_cross_lambda1
```

Use `--batch_size 1` if your GPU is shared / has <6 GB free for this process. `--lr` only affects the save-path naming when training; for `--test 1` it just needs to match the value in the OneDrive ckpt's training run so the path-encoded values line up — pass it as shown above.

### Numbers reproduced on the dev server (Aligned-orientation)

| Setting | Mean (m) ↓ — paper | Mean (m) ↓ — repro | Median (m) ↓ — paper | Median (m) ↓ — repro |
|---|---|---|---|---|
| Same λ=0   | 3.15 | **3.142** | 1.45 | **1.470** |
| Same λ=1   | 2.87 | **2.855** | 1.58 | **1.531** |
| Cross λ=0  | 3.03 | **3.033** | 1.41 | **1.418** |
| Cross λ=1  | 2.84 | **2.851** | 1.36 | **1.391** |

All 8 cells reproduce to within 0.03 m / 0.06 m.

---

## Architecture in 200 words

The model resolves the planar pose of a ground vehicle relative to a satellite tile. Ground images (perspective on KITTI, equirectangular panoramic on VIGOR) are encoded by **DINOv2 + a DPT head** (`models/dino_fit.py` + `models/dpt_single.py`) into a 32-channel feature map plus a per-pixel confidence. Each ground-image pixel is then lifted into 3D using pre-computed metric depth (`*_grd_depth.pt` on KITTI, `*_depth.npy` on VIGOR), spawning `Np = 3` Gaussian primitives whose positions, opacities, rotations and scales are predicted by `GaussianFeatEncoder` / `GaussianEncoder` (`gaussian/encoder_feat_nips.py` and `gaussian/encoder_pano.py`). The DPT feature/confidence channels are attached to each Gaussian. These Gaussians are rendered orthographically into a **128×128 BEV plane** (~101 m × 101 m on KITTI, 70 m × 70 m on VIGOR) by a custom CUDA splatter (`feature_gaussian/cuda_rasterizer/` for KITTI; `pano_feature_gaussian/cuda_rasterizer/` for VIGOR, wrapped by `vis_gaussian_feat.render_projections` / `vis_gaussian_pano.render_projections`). The BEV feature map is cross-correlated with the satellite tile's DPT features. Training uses `L = L_Weakly + λ₁ · L_GPS` — softplus on per-sample correlation peaks plus an optional consistency term inside the GPS-noise radius (paper Eq. 1).

```
ground image ─► DINOv2 ─► DPT ─► feat (32-ch) + conf
                                  │
        pre-computed depth ──────►│
                                  ▼
                        GaussianFeatEncoder           ─► Np=3 Gaussians / pixel
                                  │
                                  ▼
                        CUDA orthographic splat       ─► BEV feat (32, 128, 128)
                                  │
                                  ▼                       (cross-correlation)
sat image ─► DINOv2 ─► DPT ─► sat feat (32, 128, 128) ◄─┘
                                                          │
                                                          ▼
                                          L_Weakly + λ₁·L_GPS  (paper Eq. 1)
```

---

## Repo layout

Files you'll actually touch when reproducing or extending:

```
kitti_main/                   # ← clean entry point for KITTI Table 1
├── config.py                 # paths, image / BEV sizes, hyperparameter defaults
├── data.py                   # wraps dataLoader/KITTI_dataset
├── model.py                  # BevSplatKITTI (Stage 1 forward, mirrors models_kitti_nips.py:495-655)
├── losses.py                 # batch_wise_cross_corr + corr_for_translation + L_Weakly + L_GPS
├── eval.py                   # one evaluate() replacing legacy test1+test2
├── train.py                  # CLI + training loop
└── README.md                 # full recipe table + dropped-vs-preserved flags

vigor_main/                   # ← clean entry point for VIGOR Table 2
├── config.py                 # paths, equirectangular dims, BEV extent, lr per area
├── data.py                   # wraps dataLoader/Vigor_dataset_gs
├── model.py                  # BevSplatVIGOR (2DoF forward, mirrors models_vigor.forward2DoF)
├── losses.py                 # same loss family as KITTI, adapted to VIGOR scale convention
├── eval.py                   # one evaluate() replacing legacy test+val
├── train.py                  # CLI + training loop
└── README.md                 # recipes for the four cells

scripts/
└── bootstrap_cuda.sh         # vendor glm + build the two CUDA rasterizers in editable mode

pyproject.toml                # uv-managed Python deps, pinned to versions that produced the paper numbers
.python-version               # 3.11 — uv reads this to pick the interpreter
```

Files that contain the **algorithm itself** (touch only when changing the method):

```
gaussian/encoder_feat_nips.py     # KITTI per-pixel Gaussian primitive encoder
gaussian/encoder_pano.py          # VIGOR per-pixel Gaussian primitive encoder (equirectangular)
gaussian/build_gaussians.py       # ray casting / covariance / SH utilities
vis_gaussian_feat.py              # render_projections() — KITTI orthographic BEV splat
vis_gaussian_pano.py              # render_projections() — VIGOR orthographic BEV splat
gaussian/latent_splat_feat.py     # KITTI: render_cuda_orthographic wrapping feat_gaussian._C
gaussian/pano_splat.py            # VIGOR: render_cuda_orthographic wrapping pano_gaussian_feat._C
models/dino_fit.py                # DINOv2 + FiT-initialized trunk
models/dpt_single.py              # DPT head returning (feature, confidence)
backbone/backbone_dino_nips.py    # ResNet50 backbone used inside the KITTI encoder
backbone/backbone_pano.py         # ResNet50 backbone used inside the VIGOR encoder
feature_gaussian/                 # CUDA extension: perspective rasterizer for KITTI
pano_feature_gaussian/            # CUDA extension: panoramic rasterizer for VIGOR
dataLoader/                       # KITTI & VIGOR datasets, distance-aware batch sampler
data_utils.py, jacobian.py        # shared geometry helpers imported at top level
```

---

## Notes on the legacy code

The original training scripts (`train_KITTI_weak_nips.py`, `train_vigor_2DoF.py`) and the all-in-one model classes (`models/models_kitti_nips.py`, `models/models_vigor.py`) are left **untouched**. They still work and they're what the seq / weather / Gaussian-render-supervision experiments depend on. The recommended path for the paper's main results is `kitti_main/` and `vigor_main/`; both packages reuse the algorithm modules verbatim and only replace argparse / training loop / evaluation.

Drop-vs-preserve tables for every removed CLI flag live in each subpackage's own README.

If you're poking around the codebase for the first time, `CLAUDE.md` at the repo root contains a focused "what each file does" guide aimed at AI assistants (and humans reading by accident).

---

## Citation

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

---

## License

MIT — see [LICENSE](LICENSE). The two CUDA rasterizers under `feature_gaussian/` and `pano_feature_gaussian/` are forks of [Inria's 3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) and the [diff-feature-gaussian-rasterization](https://github.com/ywyue/FiT3D) variant; their original licenses (research/non-commercial) apply to those subtrees — see each subdirectory's `LICENSE.md`.
