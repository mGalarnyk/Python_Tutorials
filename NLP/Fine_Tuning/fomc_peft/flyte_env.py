"""Flyte 2 task environments: CPU for data, GPU for the paper-loop train."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import flyte
import flyte.app

try:  # Only the machine running `flyte deploy` needs the vLLM plugin; task pods do not.
    from flyteplugins.vllm import VLLMAppEnvironment
except ImportError:
    VLLMAppEnvironment = None

# Flyte resource GPU count is NVIDIA. On this MacBook Pro, request 0 so
# --local still runs; PyTorch uses MPS inside the task. Devbox: `flyte start
# devbox` here (no --gpu; Apple Silicon unsupported). On a remote NVIDIA
# RTX PRO 6000 Blackwell (96 GB): `flyte start devbox --gpu`, default 1 GPU.
# Set FLYTE_GPUS=4 only when a task should own four cards (DDP). For
# 1 GPU/job on a 4-GPU node keep FLYTE_GPUS=1 and pass --parallel jobs.
_HAS_NVIDIA = shutil.which("nvidia-smi") is not None


def requested_gpus() -> int:
    if not _HAS_NVIDIA:
        return 0
    return max(1, int(os.getenv("FLYTE_GPUS", "1")))


# Level 1 (docs): TaskEnvironment: image + resources. Tasks attach with @gpu_env.task.
# GPU image: requirements.txt, then the Mamba fast path (mamba-ssm + causal-conv1d,
# compiled for Blackwell sm_120) so Nemotron does not fall back to naive PyTorch.
# Set CUDA_ARCHS="9.0;12.0" before building to also cover H100.
gpu_image = (
    flyte.Image.from_debian_base(name="fomc-gpu", platform=("linux/amd64",))
    .with_requirements("requirements.txt")
    .with_source_file(Path(__file__).parent / "build_mamba_kernels.sh", "/opt")
    .with_commands([f"CUDA_ARCHS='{os.getenv('CUDA_ARCHS', '12.0')}' bash /opt/build_mamba_kernels.sh"])
)

# Flyte devbox on WSL2 (FLYTE_WSL_GPU=1 when submitting): the NVIDIA k8s device
# plugin cannot discover WSL GPUs (NVML "Not Supported"), so Kubernetes has no
# nvidia.com/gpu to schedule. GPU tasks instead use the `wsl-gpu` PodTemplate
# (devbox/wsl-gpu-podtemplate.yaml), which mounts /dev/dxg and the WSL driver
# libraries, and request no k8s GPU. Driver tasks re-import this module inside
# their own pods, so the flag travels to every pod as an env var.
WSL_GPU = os.getenv("FLYTE_WSL_GPU") == "1"
POD_ENV = {"FLYTE_WSL_GPU": "1"} if WSL_GPU else None


gpu_env = flyte.TaskEnvironment(
    name="fomc-roberta-gpu",
    image=gpu_image,
    pod_template="wsl-gpu" if WSL_GPU else None,
    env_vars=POD_ENV,
    resources=flyte.Resources(
        cpu=4,
        memory="24Gi",
        gpu=0 if WSL_GPU else requested_gpus(),
    ),
    description="Combined-S (RoBERTa + Nemotron). Same env locally (gpu=0, MPS) and on an NVIDIA CUDA GPU.",
)

# Unused. Nemotron-3-Nano-4B is another --model_key on gpu_env, not a second env.
llm_8b_env = flyte.TaskEnvironment(
    name="llm-8b-gpu",
    image=flyte.Image.from_debian_base(),
    resources=flyte.Resources(
        cpu=8,
        memory="64Gi",
        gpu=requested_gpus(),
    ),
    description="Unused. Nemotron uses gpu_env on the same Combined-S table.",
)

# Slim image so `flyte run` on the local devbox can write a Report tab
# without building the torch GPU image. Same HTML as train_one / pipeline.
ui_env = flyte.TaskEnvironment(
    name="fomc-roberta-ui",
    env_vars=POD_ENV,
    image=flyte.Image.from_debian_base(),
    resources=flyte.Resources(cpu=1, memory="1Gi"),
)

cpu_env = flyte.TaskEnvironment(
    name="fomc-roberta-cpu",
    env_vars=POD_ENV,
    image=flyte.Image.from_debian_base().with_pip_packages(
        "pandas>=2.0.0",
        "openpyxl>=3.1.0",
        "python-dotenv",
    ),
    resources=flyte.Resources(cpu=2, memory="4Gi"),
    depends_on=[gpu_env, ui_env],
)

# Serving: vLLM behind an OpenAI-compatible API, as a Flyte App. The weights
# are whatever `merge_adapter` (workflow.py) last returned: the LoRA / QLoRA
# adapter folded into the BF16 Nemotron base. Deploy after a merge run:
#   flyte deploy workflow.py vllm_app
vllm_app = None
if VLLMAppEnvironment is not None:
    vllm_app = VLLMAppEnvironment(
        name="fomc-nemotron-vllm",
        model_path=flyte.app.RunOutput(type="directory", task_name="fomc-roberta-gpu.merge_adapter"),
        model_id="fomc-nemotron",
        resources=flyte.Resources(cpu=4, memory="24Gi", gpu=1),
        extra_args=["--max-model-len", "512", "--trust-remote-code"],
        scaling=flyte.app.Scaling(replicas=(0, 1), scaledown_after=600),
        requires_auth=True,
    )
