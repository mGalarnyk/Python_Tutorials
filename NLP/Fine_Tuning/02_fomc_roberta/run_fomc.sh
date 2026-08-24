#!/usr/bin/env bash
# Train on the GPU in this shell. No Flyte, no Docker, no --local.
#
#   bash run_fomc.sh              # 18-cell CUDA grid (base+large × full/LoRA/QLoRA × 3 seeds)
#   bash run_fomc.sh qlora        # QLoRA only (6 cells). Leaves finished full/LoRA alone.
#   bash run_fomc.sh jobs         # 1 GPU/job: unfinished RoBERTa cells, one process per card
#   bash run_fomc.sh nemotron     # 1 GPU/job: Nemotron-3-Nano-4B full/LoRA/QLoRA × 3 seeds
#   bash run_fomc.sh pipeline     # one cell: roberta-base LoRA seed 5768
#
# Must be a GPU node. Login nodes are rejected.

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
  echo "[fomc] no GPU. Open a terminal on the GPU job (atl1-…), not a login node." >&2
  exit 1
fi

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
    python run_parallel.py --models nemotron-nano-4b --methods full,lora,qlora
    ;;
  grid|grid-jobs)
    python run_grid.py --no-flyte --include-qlora
    ;;
  *)
    echo "usage: bash run_fomc.sh [pipeline|grid|qlora|jobs|nemotron]" >&2
    exit 2
    ;;
esac
