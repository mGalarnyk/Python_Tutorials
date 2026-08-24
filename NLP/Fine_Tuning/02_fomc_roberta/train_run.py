"""One FOMC fine-tune run: full, LoRA, or QLoRA, with checkpoints and progress logs."""

from __future__ import annotations

import gc
import json
import os
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

import config as cfg
from device_utils import (
    describe,
    detect_device,
    empty_cache,
    from_pretrained_kwargs,
    model_dtype,
    peak_memory_gb,
    place_model,
    require_qlora,
    reset_peak_memory,
    trainer_precision,
)
from load_data import LABEL_NAMES, load_splits


def log(msg: str) -> None:
    print(f"[fomc] {msg}", flush=True)


def _quiet_transformers() -> None:
    import logging

    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("peft").setLevel(logging.ERROR)


def count_params(model) -> tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "weighted_f1": float(f1_score(labels, preds, average="weighted")),
        "macro_f1": float(f1_score(labels, preds, average="macro")),
    }


def metrics_from_preds(labels, preds) -> dict[str, Any]:
    out = {
        "accuracy": float(accuracy_score(labels, preds)),
        "weighted_f1": float(f1_score(labels, preds, average="weighted")),
        "macro_f1": float(f1_score(labels, preds, average="macro")),
    }
    per_class = f1_score(labels, preds, average=None, labels=[0, 1, 2])
    for i, name in LABEL_NAMES.items():
        out[f"f1_{name}"] = float(per_class[i])
    return out


class ProgressCallback(TrainerCallback):
    """Prints a short status line and appends JSONL so a run can be tailed."""

    def __init__(self, progress_path: Path, run_name: str):
        self.progress_path = progress_path
        self.run_name = run_name
        self.progress_path.parent.mkdir(parents=True, exist_ok=True)
        self._t0 = time.perf_counter()

    def _write(self, event: str, **payload: Any) -> None:
        rec = {
            "event": event,
            "run": self.run_name,
            "elapsed_sec": round(time.perf_counter() - self._t0, 1),
            **payload,
        }
        with self.progress_path.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")

    def on_train_begin(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        total = state.max_steps or 0
        log(f"{self.run_name}: training start  max_steps={total}  epochs={args.num_train_epochs}")
        self._write("train_begin", max_steps=total, epochs=args.num_train_epochs)

    def on_log(self, args, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        logs = logs or {}
        if "loss" not in logs and "eval_weighted_f1" not in logs:
            return
        step = state.global_step
        total = state.max_steps or 0
        pct = (100.0 * step / total) if total else 0.0
        if "loss" in logs:
            msg = (
                f"{self.run_name}: step {step}/{total} ({pct:.0f}%)  "
                f"loss={logs['loss']:.4f}"
            )
            if "learning_rate" in logs:
                msg += f"  lr={logs['learning_rate']:.2e}"
            log(msg)
            self._write("step", step=step, max_steps=total, pct=round(pct, 1), **logs)
        if "eval_weighted_f1" in logs:
            log(
                f"{self.run_name}: epoch eval  "
                f"weighted_f1={logs['eval_weighted_f1']:.4f}  "
                f"acc={logs.get('eval_accuracy', float('nan')):.4f}  "
                f"loss={logs.get('eval_loss', float('nan')):.4f}"
            )
            self._write("eval", step=step, **logs)

    def on_save(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        log(f"{self.run_name}: saved checkpoint at step {state.global_step}")
        self._write("checkpoint", step=state.global_step)

    def on_train_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        log(f"{self.run_name}: training finished after {state.global_step} steps")
        self._write("train_end", step=state.global_step)


def _is_complete(metrics_path: Path) -> bool:
    if not metrics_path.exists():
        return False
    try:
        data = json.loads(metrics_path.read_text())
    except json.JSONDecodeError:
        return False
    return data.get("status") == "complete"


def _load_model_and_tokenizer(
    model_id: str,
    method: str,
    num_labels: int = 3,
):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    _quiet_transformers()
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, do_lower_case=True, do_basic_tokenize=True
    )
    quantized = method == "qlora"

    if quantized:
        require_qlora()
        from transformers import BitsAndBytesConfig

        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float32,
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            model_id,
            num_labels=num_labels,
            quantization_config=bnb,
            device_map="auto",
        )
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(model)
    else:
        # Paper script: fp32, .to(device), no device_map.
        model = AutoModelForSequenceClassification.from_pretrained(
            model_id,
            num_labels=num_labels,
            dtype=torch.float32,
        )

    if method in {"lora", "qlora"}:
        from peft import LoraConfig, TaskType, get_peft_model

        lora = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=cfg.LORA_R,
            lora_alpha=cfg.LORA_ALPHA,
            lora_dropout=cfg.LORA_DROPOUT,
            target_modules=list(cfg.LORA_TARGET_MODULES),
            modules_to_save=list(cfg.LORA_MODULES_TO_SAVE),
        )
        model = get_peft_model(model, lora)
        model.print_trainable_parameters()

    return model, tokenizer


def _hf_dataset(df, tokenizer):
    from datasets import Dataset

    ds = Dataset.from_pandas(df[["text", "label"]], preserve_index=False)

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=cfg.MAX_LENGTH,
        )

    return ds.map(tokenize, batched=True, remove_columns=["text"])


def load_trained(
    run_directory: str | Path,
    base_model: str | None = None,
):
    """Reload a finished run (full weights or LoRA adapters) for inference."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    _quiet_transformers()
    run_directory = Path(run_directory)
    final_dir = run_directory / "final"
    src = final_dir if final_dir.exists() else run_directory
    adapter_cfg = src / "adapter_config.json"
    tokenizer = AutoTokenizer.from_pretrained(src)

    if adapter_cfg.exists():
        from peft import PeftModel

        adapter = json.loads(adapter_cfg.read_text())
        base_id = base_model or adapter.get("base_model_name_or_path")
        if not base_id:
            raise ValueError(f"No base model recorded in {adapter_cfg}")
        base = AutoModelForSequenceClassification.from_pretrained(
            base_id, num_labels=3, **from_pretrained_kwargs(model_dtype())
        )
        model = PeftModel.from_pretrained(base, str(src))
        model = place_model(model)
        return model, tokenizer

    model = AutoModelForSequenceClassification.from_pretrained(
        str(src), **from_pretrained_kwargs(model_dtype())
    )
    return place_model(model), tokenizer


def predict_texts(model, tokenizer, texts: list[str]) -> list[dict[str, Any]]:
    import torch

    model.eval()
    device = next(model.parameters()).device
    encoded = tokenizer(
        texts,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=cfg.MAX_LENGTH,
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}
    with torch.no_grad():
        logits = model(**encoded).logits
        pred_ids = logits.argmax(dim=-1).tolist()
        probs = torch.softmax(logits, dim=-1).cpu().tolist()
    out = []
    for text, pred, prob in zip(texts, pred_ids, probs):
        out.append(
            {
                "text": text,
                "pred_id": int(pred),
                "pred_name": LABEL_NAMES[int(pred)],
                "probs": {LABEL_NAMES[i]: float(p) for i, p in enumerate(prob)},
            }
        )
    return out


def evaluate_split(model, tokenizer, df) -> dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    encoded = tokenizer(
        df["text"].tolist(),
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=cfg.MAX_LENGTH,
    )
    labels = torch.tensor(df["label"].tolist(), dtype=torch.long)
    dataset = TensorDataset(encoded["input_ids"], encoded["attention_mask"], labels)
    loader = DataLoader(dataset, batch_size=16)
    device = next(model.parameters()).device
    model.eval()
    all_preds: list[int] = []
    all_labels: list[int] = []
    with torch.no_grad():
        for input_ids, attention_mask, batch_labels in loader:
            out = model(
                input_ids=input_ids.to(device),
                attention_mask=attention_mask.to(device),
            )
            all_preds.extend(out.logits.argmax(dim=-1).cpu().tolist())
            all_labels.extend(batch_labels.tolist())
    return metrics_from_preds(all_labels, all_preds)


def run_one(
    model_key: str = "roberta-base",
    method: str = "lora",
    seed: int = cfg.DEFAULT_SEED,
    split: str = cfg.DEFAULT_SPLIT,
    max_epochs: int | None = None,
    batch_size: int | None = None,
    lr: float | None = None,
    resume: bool = True,
    force: bool = False,
) -> dict[str, Any]:
    """Train one cell with the paper's loop. Skips or resumes by default."""
    from paper_loop import train_paper_loop

    if model_key not in cfg.MODELS:
        raise ValueError(f"Unknown model_key {model_key!r}. Choose from {list(cfg.MODELS)}")
    if method not in {"full", "lora", "qlora"}:
        raise ValueError("method must be full, lora, or qlora")

    model_id = cfg.MODELS[model_key]
    out_dir = cfg.run_dir(model_key, method, seed, split)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.json"
    final_dir = out_dir / "final"
    run_name = cfg.run_id(model_key, method, seed, split)

    if _is_complete(metrics_path) and not force:
        log(f"{run_name}: already complete — loading {metrics_path}")
        return json.loads(metrics_path.read_text())

    if method == "qlora":
        require_qlora()

    hyps = cfg.paper_hparams(model_key)
    max_epochs = cfg.MAX_EPOCHS if max_epochs is None else max_epochs
    batch_size = hyps["batch_size"] if batch_size is None else batch_size
    lr = hyps["lr"] if lr is None else lr

    log("=" * 64)
    log(f"run={run_name}")
    log(
        f"recipe=paper_loop  device={describe()}  method={method}  "
        f"lr={lr}  batch={batch_size}  epochs≤{max_epochs}"
    )
    log(f"checkpoints={out_dir}")
    log("=" * 64)

    train_df, test_df = load_splits(seed=str(seed), split=split)
    log(f"data: train={len(train_df):,}  test={len(test_df):,}  (val = 20% of train, torch split)")

    reset_peak_memory()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_id, do_lower_case=True, do_basic_tokenize=True
    )
    loop_out = train_paper_loop(
        build_model=lambda: _load_model_and_tokenizer(model_id, method)[0],
        tokenizer=tokenizer,
        train_df=train_df,
        test_df=test_df,
        seed=int(seed),
        batch_size=int(batch_size),
        lr=float(lr),
        out_dir=out_dir,
        run_name=run_name,
        max_epochs=max_epochs,
        resume=resume and not force,
    )

    trainable = loop_out["trainable_params"]
    total = loop_out["total_params"]
    gc.collect()
    empty_cache()

    test_metrics = loop_out["test"]
    paper = cfg.paper_f1(model_key, split)
    mem = peak_memory_gb()
    result = {
        "status": "complete",
        "run_id": run_name,
        "model": model_key,
        "method": method,
        "seed": int(seed),
        "split": split,
        "device": loop_out["device"],
        "recipe": "paper_loop",
        "lr": lr,
        "batch_size": batch_size,
        "max_epochs": max_epochs,
        "epochs_trained": loop_out["epochs_trained"],
        "best_val_weighted_f1": loop_out["best_val_weighted_f1"],
        "best_val_ce": loop_out["best_val_ce"],
        "best_val_accuracy": loop_out["best_val_accuracy"],
        "trainable_params": trainable,
        "total_params": total,
        "trainable_pct": round(100 * trainable / max(total, 1), 3),
        "peak_mem_gb": None if mem is None else round(mem, 3),
        "wall_seconds": loop_out["wall_seconds"],
        "test": test_metrics,
        "paper_weighted_f1": paper,
        "delta_vs_paper": None
        if paper is None
        else round(test_metrics["weighted_f1"] - paper, 4),
        "checkpoint_dir": str(out_dir),
        "final_dir": str(final_dir),
        "notes": loop_out.get("notes"),
    }
    metrics_path.write_text(json.dumps(result, indent=2))
    log(
        f"{run_name}: TEST weighted F1={test_metrics['weighted_f1']:.4f}  "
        f"acc={test_metrics['accuracy']:.4f}  "
        f"paper={paper}  Δ={result['delta_vs_paper']}  "
        f"time={loop_out['wall_seconds'] / 60:.1f} min"
    )
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train one FOMC RoBERTa cell")
    parser.add_argument("--model", default="roberta-base", choices=list(cfg.MODELS))
    parser.add_argument("--method", default="lora", choices=["full", "lora", "qlora"])
    parser.add_argument("--seed", type=int, default=cfg.DEFAULT_SEED)
    parser.add_argument("--split", default=cfg.DEFAULT_SPLIT)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    run_one(
        model_key=args.model,
        method=args.method,
        seed=args.seed,
        split=args.split,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        resume=not args.no_resume,
        force=args.force,
    )
