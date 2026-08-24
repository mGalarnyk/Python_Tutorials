"""Pick CUDA, Apple Silicon MPS, or CPU so the FOMC grid runs on a laptop or a GPU."""

from __future__ import annotations

import shutil

import torch


def detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def has_nvidia() -> bool:
    return torch.cuda.is_available() or shutil.which("nvidia-smi") is not None


def model_dtype() -> torch.dtype:
    """Weights dtype. Mixed-precision Trainer flags stay CUDA-only."""
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if detect_device() == "mps":
        return torch.float32
    return torch.float32


def trainer_precision() -> dict[str, bool]:
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = torch.cuda.is_available() and not use_bf16
    return {"bf16": use_bf16, "fp16": use_fp16}


def from_pretrained_kwargs(dtype: torch.dtype, quantized: bool = False) -> dict:
    if quantized:
        return {"dtype": dtype, "device_map": "auto"}
    if detect_device() == "cuda":
        return {"dtype": dtype, "device_map": "auto"}
    return {"dtype": dtype}


def place_model(model, quantized: bool = False):
    if quantized:
        return model
    device = detect_device()
    if device == "cuda":
        return model
    return model.to(device)


def require_qlora() -> None:
    if torch.cuda.is_available():
        return
    raise RuntimeError(
        "QLoRA (bitsandbytes 4-bit) needs an NVIDIA GPU. "
        f"This machine is '{detect_device()}'. Use method='lora' or method='full' instead."
    )


def empty_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif detect_device() == "mps":
        torch.mps.empty_cache()


def peak_memory_gb() -> float | None:
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024**3)
    if detect_device() == "mps":
        try:
            return torch.mps.current_allocated_memory() / (1024**3)
        except Exception:
            return None
    return None


def reset_peak_memory() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def describe() -> str:
    device = detect_device()
    extra = ""
    if device == "cuda":
        n = torch.cuda.device_count()
        name = torch.cuda.get_device_name(0)
        extra = f" ×{n} ({name})" if n > 1 else f" ({name})"
    return f"{device}{extra}"
