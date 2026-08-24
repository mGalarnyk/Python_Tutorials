"""Shah et al. published numbers, plus their released Hugging Face checkpoint."""

from __future__ import annotations

from typing import Any

import pandas as pd

import config as cfg
from load_data import LABEL_NAMES, load_splits
from train_run import evaluate_split, log


def paper_table(split: str = cfg.DEFAULT_SPLIT) -> pd.DataFrame:
    rows = []
    for model_key in cfg.MODELS:
        f1 = cfg.PAPER_WEIGHTED_F1.get((model_key, split))
        std = cfg.PAPER_WEIGHTED_F1_STD.get((model_key, split))
        rows.append(
            {
                "who": "Shah et al. (full FT, 3-seed mean)",
                "model": model_key,
                "method": "full",
                "split": split,
                "weighted_f1": f1,
                "weighted_f1_std": std,
            }
        )
    return pd.DataFrame(rows)


def eval_released_checkpoint(
    split: str = cfg.DEFAULT_SPLIT,
    seed: int = cfg.DEFAULT_SEED,
) -> dict[str, Any]:
    """Score gtfintechlab/FOMC-RoBERTa on the same test split we train against."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from device_utils import from_pretrained_kwargs, model_dtype, place_model

    log(f"loading released checkpoint {cfg.RELEASED_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.RELEASED_MODEL, do_lower_case=True, do_basic_tokenize=True
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.RELEASED_MODEL,
        num_labels=3,
        **from_pretrained_kwargs(model_dtype()),
    )
    model = place_model(model)
    _, test = load_splits(seed=str(seed), split=split)
    metrics = evaluate_split(model, tokenizer, test)
    paper = cfg.paper_f1("roberta-large", split)
    result = {
        "status": "complete",
        "who": "Shah et al. released checkpoint",
        "model": "roberta-large",
        "method": "released",
        "hf_id": cfg.RELEASED_MODEL,
        "seed": int(seed),
        "split": split,
        "test": metrics,
        "paper_weighted_f1": paper,
        "delta_vs_paper": None
        if paper is None
        else round(metrics["weighted_f1"] - paper, 4),
        "n_test": len(test),
        "label_names": LABEL_NAMES,
    }
    log(
        f"released checkpoint: TEST weighted F1={metrics['weighted_f1']:.4f}  "
        f"acc={metrics['accuracy']:.4f}  paper large={paper}  "
        "(not a matched split — their weights were not trained on this seed's train file)"
    )
    return result
