"""Flyte 2 task environments: where each task runs and what it gets.

In Flyte 2 there is no @workflow. You declare a TaskEnvironment (a container
image + CPU/GPU/memory), then decorate plain Python functions with
@env.task in workflow.py. A task that calls another task is the "pipeline".

Three environments here:
    cpu_env  data prep and driver tasks (prepare_data, pipeline, grid, release)
    gpu_env  training and merging (train_one, merge_adapter)
    ui_env   lightweight report tasks (main, snapshot_grid, write_talk_report)
plus vllm_app, a Flyte App that serves the fine-tuned model.

With `flyte run --local` every task runs in this Python process and the image
and resources below are ignored. They apply on a devbox or a cluster, where
each task runs in its own container.
"""

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

# GPUs to request for GPU tasks on a devbox / cluster. Flyte GPUs mean NVIDIA,
# so a machine without nvidia-smi (a Mac) requests 0 and PyTorch uses MPS.
_HAS_NVIDIA = shutil.which("nvidia-smi") is not None


def requested_gpus() -> int:
    if not _HAS_NVIDIA:
        return 0
    return max(1, int(os.getenv("FLYTE_GPUS", "1")))


# flyte.Image describes the container a task runs in on a devbox / cluster. Flyte
# builds it for you from these steps: a Debian base, then requirements.txt, then
# the Mamba fast path (mamba-ssm + causal-conv1d, compiled for Blackwell sm_120)
# so Nemotron does not fall back to a slow pure-PyTorch path.
# Set CUDA_ARCHS="9.0;12.0" before building to also cover H100.
gpu_image = (
    flyte.Image.from_debian_base(name="fomc-gpu", platform=("linux/amd64",))
    .with_requirements("requirements.txt")
    .with_source_file(Path(__file__).parent / "build_mamba_kernels.sh", "/opt")
    .with_commands([f"CUDA_ARCHS='{os.getenv('CUDA_ARCHS', '12.0')}' bash /opt/build_mamba_kernels.sh"])
)

# Flyte devbox on WSL2 (FLYTE_WSL_GPU=1 when submitting): Kubernetes cannot
# discover a WSL GPU (NVML "Not Supported"), so it has no nvidia.com/gpu to
# hand out. GPU tasks instead use the `wsl-gpu` PodTemplate
# (devbox/wsl-gpu-podtemplate.yaml), which mounts /dev/dxg and the WSL driver
# libraries, and request no Kubernetes GPU. Every task pod re-imports this
# module, so the flag is passed to each pod as an env var.
WSL_GPU = os.getenv("FLYTE_WSL_GPU") == "1"
POD_ENV = {"FLYTE_WSL_GPU": "1"} if WSL_GPU else None


# GPU tasks (@gpu_env.task in workflow.py): RoBERTa and Nemotron training, adapter merge.
gpu_env = flyte.TaskEnvironment(
    name="fomc-roberta-gpu",                       # shows up in the Flyte UI and task names
    image=gpu_image,                               # the container above
    pod_template="wsl-gpu" if WSL_GPU else None,   # WSL2 devbox only (see above)
    env_vars=POD_ENV,                              # environment variables set inside the task
    resources=flyte.Resources(                     # what each task asks the cluster for
        cpu=4,
        memory="24Gi",
        gpu=0 if WSL_GPU else requested_gpus(),
    ),
    description="Combined-S (RoBERTa + Nemotron). Same env locally and on an NVIDIA CUDA GPU.",
)

# Report-only tasks: a slim image (no torch) so `flyte run main` on a devbox can
# publish the comparison report without building the GPU image.
ui_env = flyte.TaskEnvironment(
    name="fomc-roberta-ui",
    env_vars=POD_ENV,
    image=flyte.Image.from_debian_base(),
    resources=flyte.Resources(cpu=1, memory="1Gi"),
)

# Data prep and driver tasks: CPU only, small image with pandas for the xlsx files.
cpu_env = flyte.TaskEnvironment(
    name="fomc-roberta-cpu",
    env_vars=POD_ENV,
    image=flyte.Image.from_debian_base().with_pip_packages(
        "pandas>=2.0.0",
        "openpyxl>=3.1.0",
        "python-dotenv",
    ),
    resources=flyte.Resources(cpu=2, memory="4Gi"),
    # Tasks in cpu_env call tasks in gpu_env and ui_env, so deploying cpu_env
    # also deploys (and builds images for) those two.
    depends_on=[gpu_env, ui_env],
)

# Serving: vLLM behind an OpenAI-compatible API, as a Flyte App (a long-running
# service, not a task that finishes). It serves whatever directory the latest
# `merge_adapter` run returned: the LoRA / QLoRA adapter folded into the BF16
# Nemotron base. Deploy after a merge run:
#   flyte deploy workflow.py vllm_app
vllm_app = None
if VLLMAppEnvironment is not None:
    vllm_app = VLLMAppEnvironment(
        name="fomc-nemotron-vllm",
        # Model weights = the output of the last merge_adapter run.
        model_path=flyte.app.RunOutput(type="directory", task_name="fomc-roberta-gpu.merge_adapter"),
        model_id="fomc-nemotron",                  # the name clients pass as "model"
        resources=flyte.Resources(cpu=4, memory="24Gi", gpu=1),
        extra_args=["--max-model-len", "512", "--trust-remote-code"],  # passed to `vllm serve`
        # Scale to zero: no replica (and no GPU) when idle; shut down after 10 idle minutes.
        scaling=flyte.app.Scaling(replicas=(0, 1), scaledown_after=600),
        requires_auth=True,                        # callers need a Flyte / Union API key
    )
