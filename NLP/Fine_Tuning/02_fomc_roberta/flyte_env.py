"""Flyte 2 task environments: CPU for data, GPU for the paper-loop train."""

from __future__ import annotations

import os
import shutil

import flyte

# Flyte resource GPU count is NVIDIA. On this MacBook Pro, request 0 so
# --local still runs; PyTorch uses MPS inside the task. Devbox: `flyte start
# devbox` here (no --gpu — Apple Silicon unsupported). On a remote NVIDIA
# RTX PRO 6000 Blackwell (96 GB): `flyte start devbox --gpu`, default 1 GPU.
# Set FLYTE_GPUS=4 only when a task should own four cards (DDP). For
# 1 GPU/job on a 4-GPU node keep FLYTE_GPUS=1 and pass --parallel jobs.
_HAS_NVIDIA = shutil.which("nvidia-smi") is not None


def requested_gpus() -> int:
    if not _HAS_NVIDIA:
        return 0
    return max(1, int(os.getenv("FLYTE_GPUS", "1")))


# Level 1 (docs): TaskEnvironment — image + resources. Tasks attach with @gpu_env.task.
gpu_env = flyte.TaskEnvironment(
    name="fomc-roberta-gpu",
    image=flyte.Image.from_debian_base().with_requirements("requirements.txt"),
    resources=flyte.Resources(
        cpu=4,
        memory="24Gi",
        gpu=requested_gpus(),
    ),
    description="RoBERTa Combined-S (full / LoRA / QLoRA). Same env locally (gpu=0, MPS) and remotely.",
)

# Placeholder only — do not attach tasks or depend_on this until we start part 3.
# Same GPU host later: Llama-3.1-8B full / LoRA / QLoRA in NLP/Fine_Tuning/03_llm_8b.
llm_8b_env = flyte.TaskEnvironment(
    name="llm-8b-gpu",
    image=flyte.Image.from_debian_base(),
    resources=flyte.Resources(
        cpu=8,
        memory="64Gi",
        gpu=requested_gpus(),
    ),
    description="Placeholder for Llama-3.1-8B later. No tasks yet.",
)

# Slim image so `flyte run` on the local devbox can write a Report tab
# without building the torch GPU image. Same HTML as train_one / pipeline.
ui_env = flyte.TaskEnvironment(
    name="fomc-roberta-ui",
    image=flyte.Image.from_debian_base(),
    resources=flyte.Resources(cpu=1, memory="1Gi"),
)

cpu_env = flyte.TaskEnvironment(
    name="fomc-roberta-cpu",
    image=flyte.Image.from_debian_base().with_pip_packages(
        "pandas>=2.0.0",
        "openpyxl>=3.1.0",
        "python-dotenv",
    ),
    resources=flyte.Resources(cpu=2, memory="4Gi"),
    depends_on=[gpu_env, ui_env],
)
