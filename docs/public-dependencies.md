# Public initialization dependencies

Run from the BevSplat root in the localization environment. These code revisions pin the public loading implementation; the used hubconf/DINO model files were compared with the training cache. The old cache had no Git metadata, so this is not a claim about its complete historical checkout.

```bash
mkdir -p external
git clone https://github.com/ywyue/FiT3D.git external/FiT3D
git -C external/FiT3D checkout 18200c21f047b5add8b7e796d6aab97fba25e790
git clone https://github.com/facebookresearch/dino.git external/dino
git -C external/dino checkout 7c446df5b9f45747937fb0d72314eb9f7b66930a
export BEVSPLAT_FIT3D_REPO="$PWD/external/FiT3D"
export BEVSPLAT_DINO_REPO="$PWD/external/dino"
export TORCH_HOME="$PWD/.cache/torch"
export HF_HOME="$PWD/.cache/huggingface"
unset HF_HUB_OFFLINE
```

The [asset manifest](../configs/public-assets.json) records names, URLs, sizes and full SHA256 values measured from the current training environment. Download and verify them before model construction, including VGG16 retained for state compatibility. The VIGOR panorama backbone uses **DINO ViT-B/8**, not ViT-B/16.

```bash
python - <<'PY'
import hashlib, json, os, pathlib, urllib.request
from huggingface_hub import hf_hub_download
manifest = json.loads(pathlib.Path('configs/public-assets.json').read_text())
target = pathlib.Path(os.environ['TORCH_HOME']) / 'hub/checkpoints'
target.mkdir(parents=True, exist_ok=True)
for asset in manifest['torch_hub']:
    path = target / asset['name']
    if not path.exists():
        urllib.request.urlretrieve(asset['url'], path)
    assert path.stat().st_size == asset['size'], path
    assert hashlib.sha256(path.read_bytes()).hexdigest() == asset['sha256'], path
asset = manifest['timm_hf']
path = pathlib.Path(hf_hub_download(asset['repo_id'], 'model.safetensors',
    revision=asset['revision'], cache_dir=os.environ['HF_HOME'] + '/hub'))
assert path.stat().st_size == asset['size']
assert hashlib.sha256(path.read_bytes()).hexdigest() == asset['sha256']
# This is this project's dedicated cache, not a shared user cache.
ref = path.parents[2] / 'refs/main'
ref.parent.mkdir(parents=True, exist_ok=True)
ref.write_text(asset['revision'])
PY
export HF_HUB_OFFLINE=1
```

FiT3D constructs timm's pretrained DINOv2 before loading its finetuned weights, so both assets are required even when the latter overwrites parameters. Public evaluation also needs these initialization assets before strict localization-state loading. Download failures or checksum changes must be resolved explicitly; do not silently replace pretrained initialization with random weights.
