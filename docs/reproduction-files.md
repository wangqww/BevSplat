# Files needed for reproduction

[Shared setup](../README.md#install-the-environment) · [KITTI guide](../kitti_main/README.md) · [VIGOR guide](../vigor_main/README.md)

Paths below are repository-relative unless prefixed by a dataset root. “Evaluation” means using a released localization checkpoint. “Training” additionally requires the training images/lists and initialization. Preprocessing models are only needed to create RGB/depth inputs; a localization `.pth` does not generate them.

## Shared code and initialization assets

| File or directory | Where it comes from / purpose | Needed when |
|---|---|---|
| `requirements-vigor.txt` | Localization Python dependencies; measured environment in root README | Model execution; KITTI clean-machine compatibility not fully verified |
| `scripts/bootstrap_cuda.sh`, `feature_gaussian/`, `pano_feature_gaussian/` | Build feature and panorama CUDA rasterizers | KITTI uses feature; VIGOR uses panorama |
| `external/FiT3D/` | Pinned code that constructs the DINOv2/FiT3D feature extractor | Training and evaluation |
| `external/dino/` | Pinned DINO Torch Hub code for supported local overrides | VIGOR training/evaluation; KITTI also has direct Hub calls |
| `docs/public-dependencies.md` | Exact source pins and download/hash-check commands for the five assets below | Setup |
| `configs/public-assets.json` | URL, size, full SHA256 and HF revision for initialization assets | Verify downloaded initialization bytes |
| `dino_resnet50_pretrain.pth` | DINO-pretrained ResNet50 Gaussian backbone initialization | Construction before localization checkpoint load |
| `dino_vitbase8_pretrain.pth` | DINO ViT-B/8 module initialization retained for model construction/state compatibility | Construction, even when unused by current forward |
| `dinov2_base_finetuned.pth` | FiT3D finetuned DINOv2 weights | Main feature extractor initialization |
| HF `model.safetensors` (`timm/vit_base_patch14_dinov2.lvd142m`) | timm DINOv2 base pretrained asset instantiated before FiT3D finetuned loading | Construction before localization checkpoint load |
| `vgg16-397923af.pth` | torchvision VGG16 compatibility module | Construction/state compatibility |

The four `.pth` initialization assets go under `$TORCH_HOME/hub/checkpoints/`; the safetensors asset uses the dedicated `$HF_HOME/hub` cache. A final localization checkpoint overwrites model parameters after construction, so it does not eliminate constructor downloads. KITTI still calls `facebookresearch/dino:main` in some backbones; the VIGOR local override and pinned assets alone are not a complete verified offline KITTI cache.

## KITTI inputs

| File/path | Purpose and required identity |
|---|---|
| `dataLoader/train_files.txt` | Bundled19655 training frame paths; no random replacement |
| `dataLoader/test1_files.txt` | Bundled3773 Same-Area rows: frame plus fixed shift/heading values |
| `dataLoader/test2_files.txt` | Bundled7542 Cross-Area rows: frame plus fixed shift/heading values |
| `configs/kitti-labels.sha256` | Exact bytes of those three lists; verify from repository root |
| `$KITTI_ROOT/satmap/<date>/<drive>/<frame>.png` | Original supplied satellite tile, including its center/direction/scale; not interchangeable with arbitrary imagery |
| `$KITTI_ROOT/depth_data/<date>/<drive>/image_02/data/<frame>.png` | Original rectified left-camera RGB; also used to scale camera intrinsics |
| `$KITTI_ROOT/depth_data/<date>/calib_cam_to_cam.txt` | Date-level `P_rect_02` camera intrinsics |
| `$KITTI_ROOT/depth_data/<date>/<drive>/oxts/data/<frame>.txt` | GPS for sampling and heading for satellite alignment; required even for GPS0 |
| `image_02/grd_depth/<frame>_grd_depth.pt` | Masked metric depth, float32 `(256,1024)`, finite/nonnegative; produced by the pinned depth companion |
| `image_02/grd_no_sky/<frame>.png` | Masked RGB consumed by localization; new CPU recipe derives black pixels from stored depth zeros |
| `scripts/prepare_kitti_rgb.py` | Produce depth-aligned 1024×256 PNGs from raw RGB + masked depth; refuses differing existing outputs |
| `dataLoader/KITTI_dataset.py` | Actual `root_dir`, frame path/perturbation parsing and input loading |
| `data_utils.py` | KITTI satellite pixel scale and camera/GPS shift conventions |
| `kitti_main/config.py` | Output `CKPT_ROOT` and model/training defaults; `KITTI_ROOT` here does not override the actual loader |
| `kitti_main/{train,data,model,eval}.py` | CLI/training/loading, loader wrappers, localization model and native evaluation |
| `train_KITTI_weak_nips.py` | Legacy Stage0 entry; edit its explicit GPU/output paths before pretraining |
| Stage0 `model_4.pth` | Complete Stage0 model state used to initialize compatible Stage1 keys, not just a rotation-head file; training-only requirement |
| `KITTI_no_GPS.pth`, `KITTI_GPS.pth` | Historical final localization models in the OneDrive parent; each tests both test1/test2. Their full-byte/complete-load/result verification is not included here |

`grd_height/` is an auxiliary depth-companion output; the main KITTI localization loader does not require it. Stage0 uses the same prepared dataset, so it does not bypass RGB/depth preparation.

## VIGOR inputs

| File/path | Purpose and required identity |
|---|---|
| `$VIGOR_ROOT/splits__corrected/<city>/satellite_list.txt` | Satellite filename index/order |
| `same_area_balanced_train__corrected.txt` | Same-Area source training pair/offset list; ordered 80%/20% train/validation |
| `same_area_balanced_test__corrected.txt` | Same-Area official test pair/offset list |
| `pano_label_balanced__corrected.txt` | Cross-Area list; city choice separates train/validation from official test |
| `configs/vigor-labels.sha256` |16 exact list identities across four cities; does not verify images or depth; corrected texts are not bundled |
| SliceMatch `VIGOR_corrected_labels/splits__corrected/` | [Pinned corrected-label source](https://github.com/tudelft-iv/SliceMatch/tree/73d27e95e4e808f8259a7e3a7bbf9944afbbc948/VIGOR_corrected_labels), commit `73d27e95e4e808f8259a7e3a7bbf9944afbbc948`; all 16 upstream files SHA256-verified against our manifest; no SliceMatch model weights/code required for localization |
| `$VIGOR_ROOT/<city>/satellite/*.jpg` | Target satellite images referenced by list indices |
| `$VIGOR_ROOT/<city>/panorama/*.jpg` | Original panorama; loader opens it even where the current forward does not use that branch |
| `$VIGOR_ROOT/<city>/pano_mask_sky/*.jpg` | Masked panorama used by the Gaussian encoder and UniK3D preprocessing |
| `$VIGOR_ROOT/<city>/sky_mask/*.png` | Optional diagnostic masks; not consumed by localization |
| `$VIGOR_ROOT/<city>/UniK3D_same_metric/*_depth.npy` | Same-Area metric depth, float32 `(1,320,640)` |
| `$VIGOR_ROOT/<city>/UniK3D_cross_metric/*_depth.npy` | Cross-Area metric depth; keep its protocol identity |
| `scripts/prepare_vigor_depth.py` | Portable bridge selecting corrected lists/order and producing the required metric-depth layout |
| `dataLoader/Vigor_dataset_gs.py` | Actual `VIGOR_ROOT`, city/list order, input parsing and ordered split |
| `vigor_main/config.py` | Output `VIGOR_CKPT_ROOT` and compatibility defaults; explicit current CLI settings override them |
| `vigor_main/{train,data,model,eval}.py` | CLI/strict checkpoint loading/training, split wrappers, behavior selection, native evaluation |
| `VIGOR_same_GPS0_corrected_E2_model14.pth` | Fixed final current Same-Area model without GPS loss |
| `VIGOR_same_GPS1_corrected_E2_model14.pth` | Fixed final current Same-Area model with GPS loss |
| `VIGOR_cross_GPS0_corrected_E2_model14.pth` | Fixed final current Cross-Area model without GPS loss |
| `VIGOR_cross_GPS1_corrected_E2_model14.pth` | Fixed final current Cross-Area model with GPS loss |
| `checkpoints/corrected-e2-manifest.json` | Maps the four names to area/GPS/recipe and full model SHA256 |
| `checkpoints/SHA256SUMS` | Verify the four downloaded localization checkpoint bytes in their download directory |

All four VIGOR files are in OneDrive `corrected-e2-fresh15-20261027`; KITTI/paper files are in its parent. Neither `model_14.pth` alone nor its SHA encodes runtime corrected/crop behavior. The VIGOR guide defines an independent evaluation block with all settings explicit.

## Preprocessing assets

Use compatible **separate preprocessing environments**. Localization backbone weights above are not replacements for these models.

| Code/model | Dataset and purpose | Exact identity/source |
|---|---|---|
| `external/Depth-Anything-V2/` | KITTI depth and VIGOR sky masking companion | commit `25ed77cad6983520960bde74ace90e65c6a8db41` |
| `metric_depth/mask_vigor_sky.py` | VIGOR V1 raw exact-zero sky rule; outputs masked JPEG + auxiliary PNG | Pinned companion above |
| `metric_depth/predict_KITTI_depth.py` | KITTI V1 `>0.05` mask × V2 metric depth; no RGB output | Pinned companion above; edit documented paths |
| `metric_depth/checkpoints/depth_anything_vitl14.pth` | Depth Anything **V1 ViT-L** sky model for both datasets | Upstream `LiheYoung/depth_anything_vitl14/pytorch_model.bin`, renamed locally; SHA256 `6c6a383e33e51c5fdfbf31e7ebcda943973a9e6a1cbef1564afe58d7f2e8fe63` |
| `metric_depth/checkpoints/depth_anything_v2_metric_vkitti_vitl.pth` | Depth Anything **V2 Metric VKITTI Large**, KITTI metric depth (not VIGOR depth) | HF revision `070e97e4b80e317ec1d03c19927304f4091a180b`;1341401064 bytes; provider LFS SHA256 `239b1054a369e66da2576e9a118d6d7c12d90dc8ebe609579a9a09cd8e05fe38` |
| `external/UniK3Dnew/` | VIGOR spherical metric-depth implementation | commit `83f63a31332e6998dabb2619a7c5b8358d17712a` |
| HF `lpiccinelli/unik3d-vitl` snapshot | VIGOR metric-depth model/config; keep full snapshot/cache, not just an arbitrarily renamed tensor | revision `008aa54308b8bc142f74c9c7b098833aaede06bd`; [download command](data-preparation.md#metric-depth) |

From the pinned Depth Anything companion root, these downloads create the exact local filenames expected above:

```bash
mkdir -p metric_depth/checkpoints
curl -fL https://huggingface.co/LiheYoung/depth_anything_vitl14/resolve/main/pytorch_model.bin   -o metric_depth/checkpoints/depth_anything_vitl14.pth
curl -fL https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-VKITTI-Large/resolve/070e97e4b80e317ec1d03c19927304f4091a180b/depth_anything_v2_metric_vkitti_vitl.pth   -o metric_depth/checkpoints/depth_anything_v2_metric_vkitti_vitl.pth
echo "6c6a383e33e51c5fdfbf31e7ebcda943973a9e6a1cbef1564afe58d7f2e8fe63  metric_depth/checkpoints/depth_anything_vitl14.pth" | sha256sum -c -
echo "239b1054a369e66da2576e9a118d6d7c12d90dc8ebe609579a9a09cd8e05fe38  metric_depth/checkpoints/depth_anything_v2_metric_vkitti_vitl.pth" | sha256sum -c -
```

The V2 identity above comes from pinned public Hugging Face LFS metadata. It verifies that download, not the byte identity of the author's historical KITTI depth generator. VIGOR does not need the V2 metric weight. Read each dataset guide before running the companion: its mask rules and output formats differ.

## Result files and reproducibility evidence

| File | Meaning |
|---|---|
| KITTI `test{1,2}_result.mat` / `test{1,2}_results.txt` | Your input's predictions/GT and native metrics for each test area |
| VIGOR `<area>_val_result.mat` / `<area>_val_results.txt` | Internal validation predictions/GT and metrics |
| VIGOR `<area>_result.mat` / `<area>_results.txt` | Official test predictions/GT and metrics |
| Saved `model_<index>.pth` | Model-only state; current VIGOR final index14 means epoch15, KITTI Stage1 final index9 means epoch10 |

These outputs are created by evaluation/training; they are not required downloads. Model-only files do not restore full optimizer/RNG state for exact training recovery. Preserve the code revision, complete CLI/env settings, source-list identities and generated-input identities with each result. Current public checksums cover the declared lists/assets/models, **not the entire raw/preprocessed corpus**. A full matching-corpus manifest and independent clean-machine rebuild/test remain necessary before asserting identical-dataset numerical reproduction.
