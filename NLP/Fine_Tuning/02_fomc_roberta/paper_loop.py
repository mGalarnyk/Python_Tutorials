"""Shah et al. training loop — faithful port of their ACL 2023 script.

Source:
https://github.com/gtfintechlab/fomc-hawkish-dovish/blob/main/code_model/bert_fine_tune_lm_hawkish_dovish_train_test.py

Matched on purpose:
  - RobertaTokenizerFast / RobertaForSequenceClassification, max_length 256
  - torch.manual_seed + np.random.seed, then torch random_split 80/20 of train
  - AdamW(lr), no scheduler, no grad clip (PyTorch default weight_decay=0.01)
  - pad the full train set once; dynamic Trainer padding is NOT used
  - early stop: 7 epochs, reset if val CE *or* acc *or* weighted F1 improves by 1e-2
  - evaluate the *last* weights after stop (they do not reload the best checkpoint)

LoRA / QLoRA reuse this loop. The variation is adapters (and optional 4-bit), not a
different optimizer schedule.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, TensorDataset

from device_utils import describe, detect_device, empty_cache
from load_data import LABEL_NAMES


def log(msg: str) -> None:
    print(f"[fomc] {msg}", flush=True)


def _is_quantized(model) -> bool:
    if getattr(model, "is_loaded_in_4bit", False) or getattr(model, "is_loaded_in_8bit", False):
        return True
    inner = getattr(model, "get_base_model", None)
    if callable(inner):
        base = inner()
        if base is not model:
            return _is_quantized(base)
    backbone = getattr(model, "backbone", None)
    if backbone is not None and backbone is not model:
        return _is_quantized(backbone)
    return False


def _train_device() -> torch.device:
    # bitsandbytes LinearFP4 needs a device *index*. `cuda` (index None) leaves
    # quant_state uninitialized → "Please call .cuda() or .to(device)".
    name = detect_device()
    return torch.device("cuda:0" if name == "cuda" else name)


def _place_for_train(model, device: torch.device):
    """LinearFP4 must see `.to(cuda:0)`. Older Params4bit rejects `.to` — skip that."""
    try:
        return model.to(device)
    except ValueError as exc:
        text = str(exc).lower()
        if "4-bit" in text or "8-bit" in text or "not supported" in text:
            log(f"skip model.to({device}): {exc}")
            return model
        raise


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


EPS = 1e-2
MAX_EARLY_STOPPING = 7
MAX_EPOCHS = 100
MAX_LENGTH = 256


def _emit(progress_path: Path, event: str, **payload: Any) -> None:
    rec = {"event": event, **payload, "t": time.time()}
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    with progress_path.open("a") as fh:
        fh.write(json.dumps(rec) + "\n")


def _texts_labels(df):
    sentences = df["text"].tolist()
    labels = df["label"].to_numpy()
    texts, y = [], []
    for i, sentence in enumerate(sentences):
        if isinstance(sentence, str):
            texts.append(sentence)
            y.append(int(labels[i]))
    return texts, np.array(y)


def _tensor_dataset(tokenizer, texts, labels, max_length: int = MAX_LENGTH) -> TensorDataset:
    tokens = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    return TensorDataset(
        tokens["input_ids"],
        tokens["attention_mask"],
        torch.LongTensor(labels),
    )


def _run_eval(model, loader, device, n: int) -> tuple[float, float, float, list[int], list[int]]:
    model.eval()
    ce = 0.0
    correct = 0
    actual: list[int] = []
    pred: list[int] = []
    with torch.no_grad():
        for input_ids, attention_masks, labels in loader:
            input_ids = input_ids.to(device)
            attention_masks = attention_masks.to(device)
            labels = labels.to(device)
            outputs = model(
                input_ids=input_ids, attention_mask=attention_masks, labels=labels
            )
            ce += outputs.loss.item() * input_ids.size(0)
            preds = torch.max(outputs.logits, 1)[1]
            correct += torch.sum(preds == labels).item()
            actual.extend(labels.cpu().tolist())
            pred.extend(preds.cpu().tolist())
    ce /= max(n, 1)
    acc = correct / max(n, 1)
    f1 = float(f1_score(actual, pred, average="weighted"))
    return ce, acc, f1, actual, pred


def train_paper_loop(
    build_model,
    tokenizer,
    train_df,
    test_df,
    *,
    seed: int,
    batch_size: int,
    lr: float,
    out_dir: Path,
    run_name: str,
    max_epochs: int = MAX_EPOCHS,
    resume: bool = True,
    max_length: int = MAX_LENGTH,
) -> dict[str, Any]:
    """Run their epoch loop. `build_model` is called *after* torch.manual_seed."""
    device = _train_device()
    progress_path = out_dir / "progress.jsonl"
    ckpt_path = out_dir / "loop_checkpoint.pt"
    final_dir = out_dir / "final"

    texts, y = _texts_labels(train_df)
    dataset = _tensor_dataset(tokenizer, texts, y, max_length=max_length)
    val_length = int(len(dataset) * 0.2)
    train_length = len(dataset) - val_length
    log(f"{run_name}: Train Size: {train_length}, Validation Size: {val_length}")

    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    model = build_model()
    model = _place_for_train(model, device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    log(f"{run_name}: params trainable={trainable:,} / {total:,} ({100 * trainable / max(total, 1):.2f}%)")

    train_set, val_set = torch.utils.data.random_split(
        dataset, lengths=[train_length, val_length]
    )
    loaders = {
        "train": DataLoader(train_set, batch_size=batch_size, shuffle=True),
        "val": DataLoader(val_set, batch_size=batch_size, shuffle=True),
    }

    optimizer = optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=lr)

    start_epoch = 0
    early_stopping_count = 0
    best_ce = float("inf")
    best_accuracy = float("-inf")
    best_f1 = float("-inf")

    if resume and ckpt_path.exists():
        blob = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(blob["model"])
        optimizer.load_state_dict(blob["optimizer"])
        start_epoch = int(blob["epoch"]) + 1
        early_stopping_count = int(blob["early_stopping_count"])
        best_ce = float(blob["best_ce"])
        best_accuracy = float(blob["best_accuracy"])
        best_f1 = float(blob["best_f1"])
        log(f"{run_name}: resumed loop at epoch {start_epoch}")

    log(
        f"{run_name}: paper loop  device={describe()}  lr={lr}  "
        f"batch={batch_size}  epochs≤{max_epochs}"
    )
    t0 = time.perf_counter()
    stopped_epoch = start_epoch

    for epoch in range(start_epoch, max_epochs):
        if early_stopping_count >= MAX_EARLY_STOPPING:
            break
        stopped_epoch = epoch
        for phase in ("train", "val"):
            if phase == "train":
                model.train()
                early_stopping_count += 1
                for input_ids, attention_masks, labels in loaders["train"]:
                    input_ids = input_ids.to(device)
                    attention_masks = attention_masks.to(device)
                    labels = labels.to(device)
                    optimizer.zero_grad()
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_masks,
                        labels=labels,
                    )
                    outputs.loss.backward()
                    optimizer.step()
            else:
                curr_ce, curr_accuracy, curr_f1, _, _ = _run_eval(
                    model, loaders["val"], device, len(val_set)
                )
                if curr_ce <= best_ce - EPS:
                    best_ce = curr_ce
                    early_stopping_count = 0
                if curr_accuracy >= best_accuracy + EPS:
                    best_accuracy = curr_accuracy
                    early_stopping_count = 0
                if curr_f1 >= best_f1 + EPS:
                    best_f1 = curr_f1
                    early_stopping_count = 0
                log(
                    f"{run_name}: epoch {epoch + 1}  "
                    f"val_ce={curr_ce:.4f}  val_acc={curr_accuracy:.4f}  "
                    f"val_f1={curr_f1:.4f}  early_stop={early_stopping_count}"
                )
                _emit(
                    progress_path,
                    "eval",
                    epoch=epoch + 1,
                    val_ce=curr_ce,
                    val_acc=curr_accuracy,
                    val_f1=curr_f1,
                    early_stopping_count=early_stopping_count,
                )

        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "early_stopping_count": early_stopping_count,
                "best_ce": best_ce,
                "best_accuracy": best_accuracy,
                "best_f1": best_f1,
            },
            ckpt_path,
        )

    # Test the last weights — they do not reload a best checkpoint.
    test_texts, test_y = _texts_labels(test_df)
    test_ds = _tensor_dataset(tokenizer, test_texts, test_y, max_length=max_length)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=True)
    _, _, _, actual, pred = _run_eval(model, test_loader, device, len(test_ds))
    test_metrics = metrics_from_preds(actual, pred)

    final_dir.mkdir(parents=True, exist_ok=True)
    if not _is_quantized(model):
        model.to("cpu")
    if hasattr(model, "save_pretrained"):
        model.save_pretrained(final_dir)
    else:
        torch.save(model.state_dict(), final_dir / "pytorch_model.bin")
    tokenizer.save_pretrained(final_dir)

    empty_cache()
    wall = time.perf_counter() - t0
    log(
        f"{run_name}: TEST weighted F1={test_metrics['weighted_f1']:.4f}  "
        f"acc={test_metrics['accuracy']:.4f}  epochs={stopped_epoch + 1}  "
        f"time={wall / 60:.1f} min"
    )
    return {
        "test": test_metrics,
        "best_val_ce": best_ce,
        "best_val_accuracy": best_accuracy,
        "best_val_weighted_f1": best_f1,
        "epochs_trained": float(stopped_epoch + 1),
        "wall_seconds": round(wall, 1),
        "device": describe(),
        "lr": lr,
        "batch_size": batch_size,
        "recipe": "paper_loop",
        "trainable_params": trainable,
        "total_params": total,
        "notes": "last checkpoint after early stop; no load_best",
    }
