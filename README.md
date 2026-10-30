# BevSplat

PyTorch implementation of [Resolving Height Ambiguity via Feature-Based Gaussian Primitives for Weakly-Supervised Cross-View Localization](https://arxiv.org/abs/2502.09080) (NeurIPS 2025).

Start with the shared installation, then follow one complete dataset guide. Both guides use the same order: required files, data preparation, configuration, evaluation, training, and results.

| Dataset | Models | Evaluation | Complete reproduction guide |
|---|---|---|---|
| KITTI | GPS0 / GPS1; each model tests both areas | test1: Same-Area; test2: Cross-Area | [KITTI guide](kitti_main/README.md) |
| VIGOR | Same/Cross-Area × GPS0/GPS1 | Separate internal validation and official test | [VIGOR guide](vigor_main/README.md) |

**GPS0** disables the GPS loss; **GPS1** enables it. See [required files and their roles](docs/reproduction-files.md) for every dataset input, preprocessing model, initialization asset, localization checkpoint, configuration entry and result file. Downloaded weights are enough to skip localization training; they do not remove the need for prepared data or model-construction assets.

## Reproduce the published-weight results

1. Obtain the matching **prepared dataset**, including its masked RGB, depth and fixed labels. The complete [VIGOR prepared archive](https://drive.google.com/drive/folders/1TVEwjSp-FFWL-c6hh-bMbrKgcCvX4c8Q) is available; follow [checksum and restoration instructions](docs/reproduction-files.md#prepared-dataset-downloads). The complete KITTI archive is still uploading. If you already have the matching prepared corpus, use it directly.
2. Install the shared environment and public model-construction assets below.
3. Follow the dataset guide's path configuration and weight checksum steps.
4. Run the guide's evaluation block: **KITTI uses `rotation_range=10`** and tests both areas with each of two weights; **VIGOR uses corrected/crop behavior with `rotation_range=0`** and four area/GPS-specific weights.

Using prepared data and released weights skips localization training and the Depth Anything/UniK3D/RGB-generation steps. Keep the archived historical inputs unchanged. [Expected results and weight identities](configs/reproduction-reference.json) provide the reference for all six models; [file roles](docs/reproduction-files.md) distinguish data, constructor assets and localization weights.

## Install the environment

The released KITTI and current VIGOR models were evaluated in the author's existing Linux environment with Python **3.10.13**, PyTorch **2.4.1**, torchvision **0.19.1**, and CUDA **11.8**. A complete clean-machine installation and data rebuild has not been verified. A compatible CUDA toolkit/compiler is required for the rasterizers.

```bash
git clone https://github.com/wangqww/BevSplat.git
cd BevSplat
conda create -n bevsplat python=3.10.13 -y
conda activate bevsplat
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements-vigor.txt
bash scripts/bootstrap_cuda.sh
python -c "import feat_gaussian, pano_gaussian_feat; print('extensions imported')"
```

Follow [public dependencies](docs/public-dependencies.md) to obtain the pinned FiT3D/DINO code and all five public model assets, with checksums. Then configure their paths in every new shell:

```bash
export BEVSPLAT_FIT3D_REPO="$PWD/external/FiT3D"
export BEVSPLAT_DINO_REPO="$PWD/external/dino"
export TORCH_HOME="$PWD/.cache/torch"
export HF_HOME="$PWD/.cache/huggingface"
export HF_HUB_OFFLINE=1
python -m kitti_main.train --help
python -m vigor_main.train --help
```

Set dataset/output paths using the chosen guide. KITTI's actual loader still requires a source-path edit and some KITTI backbones use Torch Hub directly; setting `KITTI_ROOT` alone is insufficient. The guide explains first-use downloads and offline-cache limitations. Use separate compatible environments for Depth Anything and UniK3D preprocessing so their installation does not replace localization dependencies.

## What has been reproduced

| Item | Current evidence | Remaining limit |
|---|---|---|
| VIGOR released models | All four fixed final checkpoints, independent validation/test endpoints and weight identities verified on the author's existing corpus; [measured test results](vigor_main/README.md#6-results-and-verification) | Regenerated masked JPEG/depth NPY files have not been compared file by file with that historical corpus |
| KITTI released models | Both weight identities, complete 989-key loading and full test1/test2 results verified with ±10° heading perturbations; [measured results](kitti_main/README.md#6-results-and-verification) | Recorded results use preserved historical inputs; the optional new RGB recipe is not a bitwise reconstruction of those inputs |
| Fresh-machine reproduction | Pinned loading/preprocessing code, asset checksums, explicit commands and file roles documented | No complete clean-machine data rebuild plus KITTI/VIGOR training/test reproduction has been performed |

These references establish successful released-checkpoint evaluation on the author's prepared data. Matching split hashes alone does not establish matching RGB/depth bytes. The Git repository contains code and manifests; the [VIGOR prepared download](docs/reproduction-files.md#prepared-dataset-downloads) provides the preserved corpus. KITTI archive delivery is still in progress. The provider/preprocessing instructions remain available as an alternative way to build inputs.

## Focused CPU checks

```bash
CUDA_VISIBLE_DEVICES= python -B -m unittest discover -s tests -p 'test_vigor_*.py' -v
CUDA_VISIBLE_DEVICES= python -B -m unittest discover -s tests -p 'test_kitti_rgb.py' -v
```

Some VIGOR checks require GPU/internal experiment controllers and skip by default. Older sequence/Oxford/scratch experiments are outside these two guides.

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
