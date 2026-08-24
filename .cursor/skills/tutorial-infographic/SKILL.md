---
name: tutorial-infographic
description: >-
  Generates 16:9 talk infographics for the NLP Fine_Tuning tutorials (Flyte
  pipeline, hawkish–dovish labels, full vs LoRA vs QLoRA). Use when creating,
  regenerating, or restyling those figures, or when the user asks to save or
  reuse the image prompt.
---

# Tutorial infographics

Read [NLP/Fine_Tuning/images/PROMPT.md](NLP/Fine_Tuning/images/PROMPT.md) and use the matching prompt verbatim (plus the shared visual rules).

## Rules

- Credit every figure: original diagram for this tutorial, plus idea credit if a paper supplied the content.
- Do not name a university cluster or campus in the figure or caption.
- Remote GPU is always **NVIDIA RTX PRO 6000 Blackwell (96 GB)**. One card or **four on one node** (16-core, 256 GB host). Never a vague “the GPU.”
- This laptop is the **16-inch MacBook Pro, Apple M4 Max, 128 GB, MPS**.
- The Flyte pipeline figure must show three cards: **MacBook Pro**, **1× RTX PRO 6000**, and **4× RTX PRO 6000**.
- Copy the PNG into the tutorial `images/` folder and list it in that folder’s `SOURCE.md`.
