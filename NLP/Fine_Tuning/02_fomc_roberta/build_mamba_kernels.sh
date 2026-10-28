#!/usr/bin/env bash
# Install the Mamba fast path for Nemotron-3-Nano: mamba-ssm (Triton kernels) and
# causal-conv1d (CUDA, built here). Without causal-conv1d, every Mamba block falls
# back to a naive PyTorch loop (~30 vs ~6.5 min/epoch on the laptop GPU).
#
# Needs torch already installed in the active environment. Builds against pip's
# CUDA nvcc, so no system CUDA toolkit or sudo. Used by run_fomc.sh (laptop) and
# by the Flyte GPU image (flyte_env.py).
#
#   CUDA_ARCHS="12.0" bash build_mamba_kernels.sh     # Blackwell (RTX PRO 5000/6000)
#   CUDA_ARCHS="9.0;12.0" bash build_mamba_kernels.sh # + Hopper (H100), slower build

set -euo pipefail

PY="${PYTHON:-python}"
ARCHS="${CUDA_ARCHS:-12.0}"

if "$PY" -c "import causal_conv1d, mamba_ssm" >/dev/null 2>&1; then
  echo "[mamba] causal-conv1d and mamba-ssm already installed"
  exit 0
fi

cuda_mm="$("$PY" -c 'import torch; v = torch.version.cuda.split("."); print(v[0] + "." + v[1])')"
cuda_major="${cuda_mm%%.*}"
echo "[mamba] torch CUDA ${cuda_mm}; building for sm ${ARCHS}"

uv pip install --python "$PY" "nvidia-cuda-nvcc==${cuda_mm}.*" "nvidia-cuda-cccl==${cuda_mm}.*" \
  "nvidia-cuda-crt==${cuda_mm}.*" "nvidia-nvvm==${cuda_mm}.*"
cuda_home="$("$PY" -c "import nvidia, pathlib; print(pathlib.Path(list(nvidia.__path__)[0]) / 'cu${cuda_major}')")"
ln -sfn lib "${cuda_home}/lib64"
ln -sf "libcudart.so.${cuda_major}" "${cuda_home}/lib/libcudart.so"

export CUDA_HOME="${cuda_home}" PATH="${cuda_home}/bin:${PATH}" TORCH_CUDA_ARCH_LIST="${ARCHS}" MAX_JOBS="${MAX_JOBS:-8}"
CAUSAL_CONV1D_FORCE_BUILD=TRUE uv pip install --python "$PY" --no-build-isolation --no-cache causal-conv1d
MAMBA_SKIP_CUDA_BUILD=TRUE uv pip install --python "$PY" --no-build-isolation mamba-ssm einops

# mamba-ssm imports its Mamba-1 CUDA extension at import time; Nemotron only uses
# the Triton Mamba-2 kernels, so make that import optional.
"$PY" - <<'EOF'
import importlib.util
from pathlib import Path

spec = importlib.util.find_spec("mamba_ssm")
path = Path(spec.origin).parent / "ops" / "selective_scan_interface.py"
text = path.read_text()
needle = "import selective_scan_cuda\n"
if "selective_scan_cuda = None" not in text and needle in text:
    path.write_text(text.replace(needle, "try:\n    import selective_scan_cuda\nexcept ImportError:\n    selective_scan_cuda = None\n", 1))
print(f"[mamba] patched {path}")
EOF
"$PY" -c "import causal_conv1d; from mamba_ssm.ops.triton.layernorm_gated import rmsnorm_fn; print('[mamba] fast path ready')"
