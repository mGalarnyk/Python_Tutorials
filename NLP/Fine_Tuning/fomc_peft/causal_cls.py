"""Causal LM classifiers for FOMC (Nemotron). Same 3-way F1 as RoBERTa."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from load_data import LABEL_NAMES


def hidden_size(config) -> int:
    for key in ("hidden_size", "n_embd", "d_model"):
        value = getattr(config, key, None)
        if value:
            return int(value)
    raise ValueError(f"No hidden size on {type(config)}")


def resolve_label_tokens(tokenizer) -> tuple[list[int], list[str]]:
    """First subtoken of each class name. Prefer a single-token encoding."""
    ids: list[int] = []
    used: list[str] = []
    for i in range(3):
        word = LABEL_NAMES[i]
        picked_id = None
        picked_str = None
        for cand in (word, f" {word}", word.capitalize(), word.upper()):
            pieces = tokenizer.encode(cand, add_special_tokens=False)
            if not pieces:
                continue
            picked_id = int(pieces[0])
            picked_str = cand
            if len(pieces) == 1:
                break
        if picked_id is None:
            raise ValueError(f"Tokenizer cannot encode label {word!r}")
        ids.append(picked_id)
        used.append(picked_str)
    return ids, used


def _unwrap(backbone):
    getter = getattr(backbone, "get_base_model", None)
    if callable(getter):
        return getter()
    return backbone


def _encoder(backbone):
    model = _unwrap(backbone)
    return getattr(model, "backbone", None) or getattr(model, "model", None) or model


def _lm_head(backbone):
    model = _unwrap(backbone)
    if hasattr(model, "get_output_embeddings"):
        head = model.get_output_embeddings()
        if head is not None:
            return head
    return getattr(model, "lm_head", None)


def _sanitize_generation_config(backbone) -> None:
    """Nemotron ships top_p=0.95 with do_sample unset. HF refuses to serialize that."""
    gen = getattr(backbone, "generation_config", None)
    if gen is None or getattr(gen, "do_sample", False):
        return
    if getattr(gen, "top_p", None) is not None:
        gen.top_p = None
    if getattr(gen, "temperature", None) is not None:
        gen.temperature = None


class LastTokenClassifier(nn.Module):
    """Pool the last non-pad token and score dovish / hawkish / neutral.

    Kept to load the first Nemotron grid (scratch linear head). New runs use
    LabelTokenClassifier.
    """

    def __init__(self, backbone, num_labels: int = 3):
        super().__init__()
        self.backbone = backbone
        self.config = backbone.config
        if hasattr(self.config, "use_cache"):
            self.config.use_cache = False
        dtype = next(backbone.parameters()).dtype
        self.classifier = nn.Linear(hidden_size(self.config), num_labels, dtype=dtype)

    def forward(self, input_ids, attention_mask=None, labels=None, **_kwargs):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        encoder = _encoder(self.backbone)
        out = encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            output_hidden_states=False,
        )
        hidden = out.last_hidden_state
        last = attention_mask.long().sum(dim=1) - 1
        last = last.clamp(min=0)
        pooled = hidden[torch.arange(hidden.size(0), device=hidden.device), last]
        logits = self.classifier(pooled.to(self.classifier.weight.dtype))
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.float(), labels)
        return SimpleNamespace(loss=loss, logits=logits)

    def save_pretrained(self, path, **kwargs):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        (path / "causal_cls.json").write_text(
            json.dumps({"num_labels": int(self.classifier.out_features), "mode": "linear"})
        )
        _sanitize_generation_config(self.backbone)
        if hasattr(self.backbone, "save_pretrained"):
            self.backbone.save_pretrained(path / "backbone", **kwargs)
        else:
            torch.save(self.backbone.state_dict(), path / "backbone.pt")
        torch.save(self.classifier.state_dict(), path / "classifier.pt")


class LabelTokenClassifier(nn.Module):
    """Score the LM-head logits of dovish / hawkish / neutral at the last token."""

    def __init__(self, backbone, label_token_ids: list[int]):
        super().__init__()
        self.backbone = backbone
        self.config = backbone.config
        if hasattr(self.config, "use_cache"):
            self.config.use_cache = False
        if len(label_token_ids) != 3:
            raise ValueError("Need 3 label token ids (dovish, hawkish, neutral)")
        self.register_buffer(
            "label_token_ids",
            torch.tensor(list(label_token_ids), dtype=torch.long),
            persistent=True,
        )

    def forward(self, input_ids, attention_mask=None, labels=None, **_kwargs):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        encoder = _encoder(self.backbone)
        out = encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            output_hidden_states=False,
        )
        hidden = out.last_hidden_state
        last = attention_mask.long().sum(dim=1) - 1
        last = last.clamp(min=0)
        pooled = hidden[torch.arange(hidden.size(0), device=hidden.device), last]
        head = _lm_head(self.backbone)
        if head is None:
            raise RuntimeError("No lm_head / output embeddings on the causal LM")
        weight = getattr(head, "weight", None)
        if weight is not None:
            pooled = pooled.to(weight.dtype)
        vocab_logits = head(pooled)
        # The buffer stays on CPU when load_trained uses device_map placement.
        logits = vocab_logits.index_select(-1, self.label_token_ids.to(vocab_logits.device))
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.float(), labels)
        return SimpleNamespace(loss=loss, logits=logits)

    def save_pretrained(self, path, **kwargs):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        (path / "causal_cls.json").write_text(
            json.dumps(
                {
                    "num_labels": 3,
                    "mode": "labeltok",
                    "label_token_ids": [int(x) for x in self.label_token_ids.tolist()],
                }
            )
        )
        _sanitize_generation_config(self.backbone)
        if hasattr(self.backbone, "save_pretrained"):
            self.backbone.save_pretrained(path / "backbone", **kwargs)
        else:
            torch.save(self.backbone.state_dict(), path / "backbone.pt")
