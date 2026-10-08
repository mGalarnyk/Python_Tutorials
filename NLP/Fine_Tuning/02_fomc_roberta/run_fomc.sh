#!/usr/bin/env bash
# Train on the GPU in this shell. No Flyte, no Docker, no --local.
#
#   bash run_fomc.sh              # 18-cell CUDA grid (base+large × full/LoRA/QLoRA × 3 seeds)
#   bash run_fomc.sh qlora        # QLoRA only (6 cells). Leaves finished full/LoRA alone.
#   bash run_fomc.sh jobs         # 1 GPU/job: unfinished RoBERTa cells, one process per card
#   bash run_fomc.sh nemotron     # 1 GPU/job: Nemotron-3-Nano-4B full/LoRA/QLoRA × 3 seeds (labeltok)
#   bash run_fomc.sh nemotron-lora # one seed: label-token LoRA r=64, accum 16, cosine, best ckpt
#   bash run_fomc.sh pipeline     # one cell: roberta-base LoRA seed 5768
#
# Needs an NVIDIA CUDA GPU; exits otherwise.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JOB="${1:-grid}"
export PATH="${HOME}/.local/bin:${PATH}"
export TOKENIZERS_PARALLELISM=false

cd "$ROOT"

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

if [[ ! -x .venv/bin/python ]]; then
  echo "[fomc] creating .venv (python 3.11)"
  uv venv .venv --python 3.11
  # shellcheck disable=SC1091
  source .venv/bin/activate
  uv pip install -r requirements.txt
else
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

echo "[fomc] host=$(hostname)"
echo "[fomc] python=$(command -v python)"
if ! python -c "import torch,sys; print('[fomc] cuda', torch.cuda.is_available(), torch.cuda.device_count(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''); sys.exit(0 if torch.cuda.is_available() else 1)"; then
  echo "[fomc] no CUDA GPU found. Run this on a machine with an NVIDIA GPU." >&2
  exit 1
fi

ensure_causal_conv1d() {
  # Mamba fast path. Without causal-conv1d, Nemotron's Mamba layers fall back to
  # a naive PyTorch loop (~30 min/epoch). Builds against pip's CUDA nvcc, so no
  # system CUDA toolkit or sudo is needed.
  if python -c "import causal_conv1d" >/dev/null 2>&1; then
    return
  fi
  echo "[fomc] building causal-conv1d (Mamba fast path, ~6 min)"
  local cuda_major
  cuda_major="$(python -c 'import torch; print(torch.version.cuda.split(".")[0] + "." + torch.version.cuda.split(".")[1])')"
  uv pip install "nvidia-cuda-nvcc==${cuda_major}.*" "nvidia-cuda-cccl==${cuda_major}.*" \
    "nvidia-cuda-crt==${cuda_major}.*" "nvidia-nvvm==${cuda_major}.*"
  local cuda_home
  cuda_home="$(python -c 'import nvidia, pathlib, torch; print(pathlib.Path(list(nvidia.__path__)[0]) / ("cu" + torch.version.cuda.split(".")[0]))')"
  ln -sfn lib "${cuda_home}/lib64"
  ln -sf libcudart.so."$(python -c 'import torch; print(torch.version.cuda.split(".")[0])')" "${cuda_home}/lib/libcudart.so"
  CUDA_HOME="${cuda_home}" PATH="${cuda_home}/bin:${PATH}" CAUSAL_CONV1D_FORCE_BUILD=TRUE MAX_JOBS=8 \
    uv pip install --no-build-isolation --no-cache causal-conv1d
}

case "$JOB" in
  pipeline)
    python train_run.py --model roberta-base --method lora --seed 5768
    ;;
  qlora)
    python run_grid.py --no-flyte --models roberta-base,roberta-large --methods qlora
    ;;
  jobs)
    python run_parallel.py
    ;;
  nemotron)
    if ! python -c "from mamba_ssm.ops.triton.layernorm_gated import rmsnorm_fn" >/dev/null 2>&1; then
      echo "[fomc] installing mamba-ssm (Nemotron hybrid import)"
      MAMBA_SKIP_CUDA_BUILD=TRUE uv pip install --no-build-isolation mamba-ssm einops
      python patch_mamba_ssm.py
    fi
    if ! python -c "from mamba_ssm.ops.triton.layernorm_gated import rmsnorm_fn"; then
      echo "[fomc] mamba-ssm still missing after install." >&2
      echo "  source .venv/bin/activate" >&2
      echo "  MAMBA_SKIP_CUDA_BUILD=TRUE uv pip install --no-build-isolation mamba-ssm einops" >&2
      echo "  python patch_mamba_ssm.py" >&2
      exit 1
    fi
    echo "[fomc] mamba-ssm ok"
    ensure_causal_conv1d
    python run_parallel.py --models nemotron-nano-4b --methods full,lora,qlora
    ;;
  nemotron-lora)
    if ! python -c "from mamba_ssm.ops.triton.layernorm_gated import rmsnorm_fn" >/dev/null 2>&1; then
      echo "[fomc] installing mamba-ssm (Nemotron hybrid import)"
      MAMBA_SKIP_CUDA_BUILD=TRUE uv pip install --no-build-isolation mamba-ssm einops
      python patch_mamba_ssm.py
    fi
    if ! python -c "from mamba_ssm.ops.triton.layernorm_gated import rmsnorm_fn"; then
      echo "[fomc] mamba-ssm still missing after install." >&2
      echo "  source .venv/bin/activate" >&2
      echo "  MAMBA_SKIP_CUDA_BUILD=TRUE uv pip install --no-build-isolation mamba-ssm einops" >&2
      echo "  python patch_mamba_ssm.py" >&2
      exit 1
    fi
    echo "[fomc] mamba-ssm ok"
    ensure_causal_conv1d
    echo "[fomc] labeltok LoRA probe: r=64  lr=1e-4  accum=16  cosine  best val-F1  seed=5768"
    python train_run.py --model nemotron-nano-4b --method lora --seed 5768
    ;;
  grid|grid-jobs)
    python run_grid.py --no-flyte --include-qlora
    ;;
  *)
    echo "usage: bash run_fomc.sh [pipeline|grid|qlora|jobs|nemotron|nemotron-lora]" >&2
    exit 2
    ;;
esac
