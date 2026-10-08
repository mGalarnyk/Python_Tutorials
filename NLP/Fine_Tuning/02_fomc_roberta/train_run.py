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


def _replace_linears_with_nf4(module, skip=("classifier",)) -> int:
    """Swap nn.Linear for LinearNF4 (unquantized). `.to(cuda:0)` quantizes later."""
    import bitsandbytes as bnb
    import torch.nn as nn

    n = 0
    for name, child in list(module.named_children()):
        if name in skip:
            continue
        if isinstance(child, bnb.nn.Linear4bit):
            continue
        if isinstance(child, nn.Linear):
            new = bnb.nn.LinearNF4(
                child.in_features,
                child.out_features,
                bias=child.bias is not None,
                compute_dtype=child.weight.dtype,
                compress_statistics=True,
            )
            new.weight = bnb.nn.Params4bit(
                child.weight.data.detach().contiguous(),
                requires_grad=False,
                compress_statistics=True,
                quant_type="nf4",
                module=new,
                bnb_quantized=False,
            )
            if child.bias is not None:
                new.bias = nn.Parameter(child.bias.data.detach().clone())
            setattr(module, name, new)
            n += 1
        else:
            n += _replace_linears_with_nf4(child, skip=skip)
    return n


def _assert_4bit_ready(model) -> None:
    import bitsandbytes as bnb

    names = []
    bad = []
    for name, mod in model.named_modules():
        if isinstance(mod, bnb.nn.Linear4bit):
            names.append(name)
            if getattr(mod.weight, "quant_state", None) is None:
                bad.append(name)
    if not names:
        raise RuntimeError("QLoRA: no Linear4bit layers after replace")
    if bad:
        raise RuntimeError(
            f"QLoRA: {len(bad)}/{len(names)} Linear4bit layers have no quant_state "
            f"(first: {bad[:6]})"
        )
    log(f"QLoRA: {len(names)} LinearNF4 layers quantized")


def _ensure_pad_token(tokenizer):
    """Causal LMs often have no pad token. Reuse EOS so paper_loop can pad."""
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is None:
            raise ValueError("Tokenizer has neither pad_token nor eos_token")
        tokenizer.pad_token = tokenizer.eos_token
        log(f"tokenizer.pad_token = eos_token ({tokenizer.eos_token!r})")
    return tokenizer


def _load_tokenizer(model_id: str, model_key: str):
    from transformers import AutoTokenizer

    _quiet_transformers()
    if cfg.is_causal(model_key):
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        tokenizer.padding_side = "right"
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            model_id, do_lower_case=True, do_basic_tokenize=True
        )
    return _ensure_pad_token(tokenizer)


def _prepare_frame(df, model_key: str, tokenizer):
    if not cfg.is_causal(model_key):
        return df
    from fomc_prompt import encode_classify

    out = df.copy()
    out["text"] = [encode_classify(text, tokenizer) for text in out["text"]]
    return out


def _quantize_on_cuda(model) -> None:
    import bitsandbytes as bnb

    model.to("cuda:0")
    for mod in model.modules():
        if not isinstance(mod, bnb.nn.Linear4bit):
            continue
        weight = mod.weight
        if getattr(weight, "quant_state", None) is not None:
            continue
        if isinstance(weight, bnb.nn.Params4bit):
            weight.bnb_quantized = False
            new_w = weight.to(device="cuda:0")
            mod.weight = new_w
            if getattr(new_w, "quant_state", None) is not None:
                mod.quant_state = new_w.quant_state
    _assert_4bit_ready(model)


def _load_model(
    model_id: str,
    model_key: str,
    method: str,
    tokenizer,
    num_labels: int = 3,
):
    import torch
    from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification

    _quiet_transformers()
    quantized = method == "qlora"
    if cfg.is_causal(model_key):
        from causal_cls import LabelTokenClassifier, resolve_label_tokens
        from peft import LoraConfig, TaskType, get_peft_model

        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        backbone = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=dtype,
            trust_remote_code=True,
        )
        if tokenizer.pad_token_id is not None:
            backbone.config.pad_token_id = tokenizer.pad_token_id
        if hasattr(backbone, "gradient_checkpointing_enable"):
            backbone.gradient_checkpointing_enable()
            log("Nemotron: gradient checkpointing on")
        if quantized:
            require_qlora()
            n = _replace_linears_with_nf4(backbone, skip=cfg.NEMOTRON_NF4_SKIP)
            log(f"QLoRA: replaced {n} Linear layers with LinearNF4 (kept BF16: {', '.join(cfg.NEMOTRON_NF4_SKIP)})")
        if method in {"lora", "qlora"}:
            targets = list(cfg.lora_target_modules(model_key))
            log(f"LoRA targets={targets}  r={cfg.lora_r(model_key)}  alpha={cfg.lora_alpha(model_key)}")
            lora = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=cfg.lora_r(model_key),
                lora_alpha=cfg.lora_alpha(model_key),
                lora_dropout=cfg.LORA_DROPOUT,
                target_modules=targets,
            )
            backbone = get_peft_model(backbone, lora)
            backbone.print_trainable_parameters()
        label_ids, label_strs = resolve_label_tokens(tokenizer)
        log(f"Nemotron label tokens={list(zip(label_strs, label_ids))}")
        model = LabelTokenClassifier(backbone, label_token_ids=label_ids)
        if quantized:
            _quantize_on_cuda(model)
        return model

    # Paper script: fp32, .to(device), no device_map. QLoRA replaces encoder
    # linears with LinearNF4 and quantizes on the CPU→GPU move — transformers'
    # BitsAndBytesConfig leaves LinearFP4 with quant_state=None on this bnb.
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        num_labels=num_labels,
        dtype=torch.float32,
    )
    if quantized:
        require_qlora()
        n = _replace_linears_with_nf4(model)
        log(f"QLoRA: replaced {n} Linear layers with LinearNF4 (classifier kept fp32)")

    if method in {"lora", "qlora"}:
        from peft import LoraConfig, TaskType, get_peft_model

        lora = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=cfg.lora_r(model_key),
            lora_alpha=cfg.lora_alpha(model_key),
            lora_dropout=cfg.LORA_DROPOUT,
            target_modules=cfg.lora_target_modules(model_key),
            modules_to_save=list(cfg.LORA_MODULES_TO_SAVE),
        )
        model = get_peft_model(model, lora)
        model.print_trainable_parameters()

    if quantized:
        _quantize_on_cuda(model)
    return model


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
    tokenizer = AutoTokenizer.from_pretrained(src, trust_remote_code=True)
    _ensure_pad_token(tokenizer)

    causal_marker = src / "causal_cls.json"
    if causal_marker.exists() or (src / "backbone").exists():
        import torch
        from transformers import AutoModelForCausalLM

        meta = {"num_labels": 3}
        if causal_marker.exists():
            meta = json.loads(causal_marker.read_text())
        backbone_dir = src / "backbone"
        if (backbone_dir / "adapter_config.json").exists():
            from peft import PeftModel

            adapter = json.loads((backbone_dir / "adapter_config.json").read_text())
            base_id = base_model or adapter.get("base_model_name_or_path")
            if not base_id:
                raise ValueError(f"No base model recorded in {backbone_dir / 'adapter_config.json'}")
            base = AutoModelForCausalLM.from_pretrained(
                base_id, trust_remote_code=True, **from_pretrained_kwargs(model_dtype())
            )
            backbone = PeftModel.from_pretrained(base, str(backbone_dir))
        else:
            load_from = str(backbone_dir if backbone_dir.exists() else src)
            backbone = AutoModelForCausalLM.from_pretrained(
                load_from, trust_remote_code=True, **from_pretrained_kwargs(model_dtype())
            )
        if meta.get("mode") == "labeltok" or meta.get("label_token_ids"):
            from causal_cls import LabelTokenClassifier

            model = LabelTokenClassifier(
                backbone, label_token_ids=[int(x) for x in meta["label_token_ids"]]
            )
        else:
            from causal_cls import LastTokenClassifier

            model = LastTokenClassifier(backbone, num_labels=int(meta.get("num_labels", 3)))
            clf_path = src / "classifier.pt"
            if clf_path.exists():
                model.classifier.load_state_dict(
                    torch.load(clf_path, map_location="cpu", weights_only=True)
                )
        return place_model(model), tokenizer

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

    _ensure_pad_token(tokenizer)
    sentences = list(texts)
    if getattr(model, "backbone", None) is not None:
        from fomc_prompt import encode_classify

        texts = [encode_classify(text, tokenizer) for text in texts]
    model.eval()
    device = next(model.parameters()).device
    encoded = tokenizer(
        texts,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=cfg.NEMOTRON_MAX_LENGTH if getattr(model, "backbone", None) is not None else cfg.MAX_LENGTH,
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}
    with torch.no_grad():
        logits = model(**encoded).logits
        pred_ids = logits.argmax(dim=-1).tolist()
        probs = torch.softmax(logits, dim=-1).cpu().tolist()
    out = []
    for text, pred, prob in zip(sentences, pred_ids, probs):
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

    texts = df["text"].tolist()
    if getattr(model, "backbone", None) is not None:
        from fomc_prompt import encode_classify

        texts = [encode_classify(text, tokenizer) for text in texts]
    _ensure_pad_token(tokenizer)
    encoded = tokenizer(
        texts,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=cfg.NEMOTRON_MAX_LENGTH if getattr(model, "backbone", None) is not None else cfg.MAX_LENGTH,
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
        cached = json.loads(metrics_path.read_text())
        test = cached.get("test") or {}
        log(
            f"DONE (cached) {model_key} {method} seed={seed}  "
            f"F1={test.get('weighted_f1', '—')}  {metrics_path}"
        )
        return cached

    if method == "qlora":
        require_qlora()

    hyps = cfg.paper_hparams(model_key, method)
    max_epochs = cfg.MAX_EPOCHS if max_epochs is None else max_epochs
    batch_size = hyps["batch_size"] if batch_size is None else batch_size
    lr = hyps["lr"] if lr is None else lr
    grad_accum = int(hyps.get("grad_accum") or 1)
    cosine = bool(hyps.get("cosine"))
    load_best = bool(hyps.get("load_best"))

    log("=" * 64)
    log(f"START {model_key} {method} seed={seed}")
    log(f"run={run_name}")
    log(
        f"recipe={cfg.recipe_for(model_key)}  device={describe()}  method={method}  "
        f"lr={lr}  batch={batch_size}  accum={grad_accum}  cosine={cosine}  "
        f"load_best={load_best}  epochs≤{max_epochs}"
    )
    log(f"checkpoints={out_dir}")
    log("=" * 64)

    train_df, test_df = load_splits(seed=str(seed), split=split)
    log(f"data: train={len(train_df):,}  test={len(test_df):,}  (val = 20% of train, torch split)")

    reset_peak_memory()

    tokenizer = _load_tokenizer(model_id, model_key)
    train_df = _prepare_frame(train_df, model_key, tokenizer)
    test_df = _prepare_frame(test_df, model_key, tokenizer)
    loop_out = train_paper_loop(
        build_model=lambda: _load_model(model_id, model_key, method, tokenizer),
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
        max_length=cfg.max_length_for(model_key),
        grad_accum=grad_accum,
        cosine=cosine,
        load_best=load_best,
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
        "recipe": loop_out.get("recipe") or cfg.recipe_for(model_key),
        "lr": lr,
        "batch_size": batch_size,
        "grad_accum": loop_out.get("grad_accum", grad_accum),
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
        f"DONE {model_key} {method} seed={seed}  "
        f"TEST weighted F1={test_metrics['weighted_f1']:.4f}  "
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
