#!/usr/bin/env bash
# Vendor glm headers and build the two CUDA rasterizers in editable mode.
#
# Assumes:
#   - A working uv venv is active OR the system Python already has torch
#     2.4.1 + nvcc 11.8 on PATH (use ``source .venv/bin/activate`` if you
#     installed the README requirements).
#   - The repo is checked out at $REPO_ROOT (auto-detected from the script
#     location).
#
# After this script finishes you should be able to do:
#     python -c "import feat_gaussian, pano_gaussian_feat; print('ok')"
#
# Re-running is safe — glm gets re-copied (no-op if identical) and pip
# rebuilds incrementally.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GLM_VERSION="0.9.9.8"
GLM_CHECKOUT="${TMPDIR:-/tmp}/bevsplat-glm-${GLM_VERSION}"

echo "[bootstrap] Repo root: ${REPO_ROOT}"

# ---------------------------------------------------------------------------
# 1. Sanity checks
# ---------------------------------------------------------------------------
command -v python  >/dev/null || { echo "[bootstrap] FATAL: python not on PATH"; exit 1; }
command -v nvcc    >/dev/null || { echo "[bootstrap] FATAL: nvcc not on PATH; install CUDA toolkit 11.8 or set PATH"; exit 1; }
command -v git     >/dev/null || { echo "[bootstrap] FATAL: git required to fetch glm headers"; exit 1; }
command -v ninja   >/dev/null || { echo "[bootstrap] WARN: ninja missing; torch.utils.cpp_extension build will be slow"; }

python -c "import torch; v = torch.__version__; assert v.split('+')[0] == '2.4.1', f'current VIGOR release requires torch 2.4.1, got {v}'; assert torch.version.cuda, 'torch CUDA build required'"
echo "[bootstrap] torch $(python -c 'import torch; print(torch.__version__)') OK"

# ---------------------------------------------------------------------------
# 2. Fetch glm headers (vendored upstream — header-only library)
# ---------------------------------------------------------------------------
if [ ! -f "${GLM_CHECKOUT}/glm/glm.hpp" ]; then
  echo "[bootstrap] Cloning glm ${GLM_VERSION} into ${GLM_CHECKOUT}"
  rm -rf "${GLM_CHECKOUT}"
  git clone --depth 1 --branch "${GLM_VERSION}" https://github.com/g-truc/glm.git "${GLM_CHECKOUT}"
else
  echo "[bootstrap] Reusing glm checkout at ${GLM_CHECKOUT}"
fi

for ext_dir in feature_gaussian pano_feature_gaussian; do
  dst="${REPO_ROOT}/${ext_dir}/third_party/glm/glm"
  mkdir -p "$(dirname "${dst}")"
  rm -rf "${dst}"
  cp -r "${GLM_CHECKOUT}/glm" "${dst}"
  echo "[bootstrap] glm vendored into ${ext_dir}/third_party/glm/glm"
done

# ---------------------------------------------------------------------------
# 3. Build & install the two CUDA rasterizers (editable)
# ---------------------------------------------------------------------------
for ext_dir in feature_gaussian pano_feature_gaussian; do
  echo "[bootstrap] Building ${ext_dir} (this takes a few minutes)"
  (
    cd "${REPO_ROOT}/${ext_dir}"
    pip install -e . --no-build-isolation
  )
done

# ---------------------------------------------------------------------------
# 4. Smoke test
# ---------------------------------------------------------------------------
python -c "import feat_gaussian, pano_gaussian_feat; print('[bootstrap] CUDA extensions OK')"
