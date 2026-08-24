"""Laptop vs GPU defaults for the Union text-to-SQL pipeline."""

from __future__ import annotations

from device_utils import detect_device, describe, has_nvidia


def recommend(smoke: bool = True) -> dict:
    """
    smoke=True: tiny run to prove the pipeline (Mac or first GPU check).
    smoke=False: closer to Union's defaults. Use on the remote RTX PRO 6000 Blackwell.
    """
    device = detect_device()
    if smoke or device != "cuda":
        return {
            "device": describe(),
            "method": "lora",
            "epochs": 1,
            "batch_size": 2 if device == "mps" else 4,
            "max_train_samples": 100,
            "max_eval_samples": 20,
            "num_eval_examples": 10,
            "qlora_ok": device == "cuda",
        }
    return {
        "device": describe(),
        "method": "lora",
        "epochs": 3,
        "batch_size": 4,
        "max_train_samples": 5000,
        "max_eval_samples": 500,
        "num_eval_examples": 50,
        "qlora_ok": True,
    }


def flyte_local_argv(profile: dict | None = None, method: str | None = None) -> list[str]:
    p = profile or recommend()
    chosen = method or p["method"]
    if chosen == "qlora" and not p["qlora_ok"] and not has_nvidia():
        raise RuntimeError("QLoRA needs NVIDIA CUDA. On this laptop use lora or full.")
    return [
        "flyte",
        "run",
        "--local",
        "workflow.py",
        "pipeline",
        "--method",
        chosen,
        "--epochs",
        str(p["epochs"]),
        "--batch_size",
        str(p["batch_size"]),
        "--max_train_samples",
        str(p["max_train_samples"]),
        "--max_eval_samples",
        str(p["max_eval_samples"]),
        "--num_eval_examples",
        str(p["num_eval_examples"]),
    ]
