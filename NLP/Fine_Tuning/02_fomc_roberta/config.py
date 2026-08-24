"""Defaults that match Shah, Paturi, and Chava (ACL 2023) as closely as we can."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent
CHECKPOINTS_DIR = ROOT / "checkpoints"
OUTPUTS_DIR = ROOT / "outputs"

# Combined-S in the paper (sentences split on but/however/…). Same seed they used first.
DEFAULT_SEED = 5768
DEFAULT_SPLIT = "lab-manual-split-combine"
PAPER_SEEDS = (5768, 78516, 944601)

# Their training loop: max 100 epochs, stop after 7 epochs with no 1e-2 gain.
MAX_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 7
EARLY_STOPPING_THRESHOLD = 0.01
VAL_FRACTION = 0.2
MAX_LENGTH = 256
WEIGHT_DECAY = 0.01

# Combined-S grid-search winners (their xlsx, pick by mean val F1):
#   roberta-base  → lr 1e-5, batch 8  → test F1 0.698
#   roberta-large → lr 1e-5, batch 16 → test F1 0.711
PAPER_FULL_HPARAMS = {
    "roberta-base": {"lr": 1e-5, "batch_size": 8},
    "roberta-large": {"lr": 1e-5, "batch_size": 16},
}

# LoRA / QLoRA reuse the paper loop and the same LR/batch so the only
# change is adapters (or 4-bit). Do not bump LoRA LR unless that is the
# variation you are measuring.
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.1
LORA_TARGET_MODULES = ["query", "key", "value", "dense"]
LORA_MODULES_TO_SAVE = ["classifier"]

MODELS = {
    "roberta-base": "roberta-base",
    "roberta-large": "roberta-large",
}

# Complete comparison: both paper sizes × all three methods.
# Mac follow-along drops QLoRA (no bitsandbytes 4-bit on MPS).
# Remote RTX PRO 6000 Blackwell runs this full list — not QLoRA-only.
ALL_JOBS = (
    ("roberta-base", "full"),
    ("roberta-base", "lora"),
    ("roberta-base", "qlora"),
    ("roberta-large", "full"),
    ("roberta-large", "lora"),
    ("roberta-large", "qlora"),
)

# Table 5 of the paper: mean weighted F1 over three seeds. Full fine-tune only.
PAPER_WEIGHTED_F1 = {
    ("roberta-base", "lab-manual-combine"): 0.6755,
    ("roberta-base", "lab-manual-split-combine"): 0.6981,
    ("roberta-large", "lab-manual-combine"): 0.7171,
    ("roberta-large", "lab-manual-split-combine"): 0.7113,
}
PAPER_WEIGHTED_F1_STD = {
    ("roberta-base", "lab-manual-combine"): 0.0267,
    ("roberta-base", "lab-manual-split-combine"): 0.0097,
    ("roberta-large", "lab-manual-combine"): 0.0164,
    ("roberta-large", "lab-manual-split-combine"): 0.0106,
}

# Released checkpoint is RoBERTa-large, full FT, Combined-S.
RELEASED_MODEL = "gtfintechlab/FOMC-RoBERTa"
RELEASED_PAPER_URL = "https://arxiv.org/abs/2305.07972"
RELEASED_CODE_URL = "https://github.com/gtfintechlab/fomc-hawkish-dovish"


RECIPE = "paper"  # faithful Shah et al. loop, not Hugging Face Trainer


def paper_hparams(model_key: str) -> dict:
    if model_key not in PAPER_FULL_HPARAMS:
        raise KeyError(f"No paper hparams for {model_key}")
    return dict(PAPER_FULL_HPARAMS[model_key])


def run_id(model_key: str, method: str, seed: int, split: str = DEFAULT_SPLIT) -> str:
    return f"{model_key}__{method}__{split}__seed{seed}__{RECIPE}"


def run_dir(model_key: str, method: str, seed: int, split: str = DEFAULT_SPLIT) -> Path:
    return CHECKPOINTS_DIR / run_id(model_key, method, seed, split)


def paper_f1(model_key: str, split: str = DEFAULT_SPLIT) -> float | None:
    return PAPER_WEIGHTED_F1.get((model_key, split))
