"""Last-token classifier on a causal LM (Nemotron). Same 3-way F1 as RoBERTa."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F


def hidden_size(config) -> int:
    for key in ("hidden_size", "n_embd", "d_model"):
        value = getattr(config, key, None)
        if value:
            return int(value)
    raise ValueError(f"No hidden size on {type(config)}")


class LastTokenClassifier(nn.Module):
    """Pool the last non-pad token and score dovish / hawkish / neutral."""

    def __init__(self, backbone, num_labels: int = 3):
        super().__init__()
        self.backbone = backbone
        self.config = backbone.config
        if hasattr(self.config, "use_cache"):
            self.config.use_cache = False
        dtype = next(backbone.parameters()).dtype
        self.classifier = nn.Linear(hidden_size(self.config), num_labels, dtype=dtype)

    def _encoder(self):
        """NemotronHModel (or equivalent) — skip the causal LM head / vocab logits."""
        model = self.backbone
        getter = getattr(model, "get_base_model", None)
        if callable(getter):
            model = getter()
        return getattr(model, "backbone", None) or getattr(model, "model", None) or model

    def forward(self, input_ids, attention_mask=None, labels=None, **_kwargs):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        encoder = self._encoder()
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
        (path / "causal_cls.json").write_text(json.dumps({"num_labels": int(self.classifier.out_features)}))
        if hasattr(self.backbone, "save_pretrained"):
            self.backbone.save_pretrained(path / "backbone", **kwargs)
        else:
            torch.save(self.backbone.state_dict(), path / "backbone.pt")
        torch.save(self.classifier.state_dict(), path / "classifier.pt")
