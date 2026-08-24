import os
import shutil
from dotenv import load_dotenv
import flyte

load_dotenv()

# Flyte resource GPU count is NVIDIA. On a Mac laptop, request 0 so --local still runs;
# PyTorch will use MPS inside the task. On a remote NVIDIA RTX PRO 6000 Blackwell,
# default 1 GPU. Keep FLYTE_GPUS=1 for 1 GPU/job; set 4 only when a task should own the node.
_HAS_NVIDIA = shutil.which("nvidia-smi") is not None


def requested_gpus() -> int:
    if not _HAS_NVIDIA:
        return 0
    return max(1, int(os.getenv("FLYTE_GPUS", "1")))


gpu_env = flyte.TaskEnvironment(
    name="llm-finetune-gpu",
    image=flyte.Image.from_debian_base().with_requirements("requirements.txt"),
    resources=flyte.Resources(
        cpu=4,
        memory="24Gi",
        gpu=requested_gpus(),
    ),
    secrets=[
        # flyte.Secret(key="HF_TOKEN", as_env_var="HF_TOKEN"),
    ],
)

cpu_env = flyte.TaskEnvironment(
    name="llm-finetune-cpu",
    image=flyte.Image.from_debian_base().with_pip_packages(
        "datasets>=3.0.0", "markdown", "python-dotenv",
    ),
    resources=flyte.Resources(cpu=2, memory="4Gi"),
    depends_on=[gpu_env],
)

HF_TOKEN = os.getenv("HF_TOKEN")
