"""FOMC label prompt for a causal LM (Nemotron). RoBERTa uses the raw sentence."""

from __future__ import annotations

LABELS = ("dovish", "hawkish", "neutral")

# Reasoning off — we want a stance token, not a long trace.
NEMOTRON_SYSTEM = (
    "You classify Federal Open Market Committee sentences. "
    "Reply with exactly one label: dovish, hawkish, or neutral. "
    "Do not explain."
)


def classify_prompt(sentence: str) -> str:
    return (
        f"{NEMOTRON_SYSTEM}\n\n"
        "Classify the FOMC sentence as dovish, hawkish, or neutral.\n"
        f"Sentence: {sentence.strip()}\n"
        "Label:"
    )


def encode_classify(sentence: str, tokenizer=None) -> str:
    """Reasoning-off classify string. Uses the chat template when the tokenizer has one."""
    text = str(sentence).strip()
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        messages = [
            {"role": "system", "content": NEMOTRON_SYSTEM},
            {
                "role": "user",
                "content": (
                    "Classify the FOMC sentence as dovish, hawkish, or neutral.\n"
                    f"Sentence: {text}"
                ),
            },
        ]
        kwargs = {"tokenize": False, "add_generation_prompt": True}
        try:
            return tokenizer.apply_chat_template(
                messages, enable_thinking=False, **kwargs
            )
        except TypeError:
            return tokenizer.apply_chat_template(messages, **kwargs)
    return classify_prompt(text)
