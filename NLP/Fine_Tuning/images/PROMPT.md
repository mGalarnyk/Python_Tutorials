# Prompt for tutorial infographics

Use this when generating a 16:9 talk figure for these LoRA / QLoRA notebooks (Flyte pipeline, hawkish–dovish labels, full vs LoRA vs QLoRA). Always credit the result as an original diagram for this tutorial, plus idea credit where the content comes from a paper.

Copy the block that matches the figure. Keep text short, high contrast, no watermarks, no misspellings, no fake screenshots.

Do not name a university cluster or campus in the figure.

## Shared visual rules

```
Clean educational infographic, 16:9 landscape, white background, for a technical talk.
No decorative photos, no 3D, no fake screenshots or fake terminal windows with unreadable text.
High contrast, readable sans-serif, generous whitespace, no watermark, no misspelled words.
Title at top in dark navy, centered. Subtitle in gray.
Rounded cards, plenty of padding. CPU vs GPU as small badges, not giant labels.
```

## `flyte_pipeline.png` (part 2 — FOMC)

Card washes must match the Results table: Mac `#dbeafe`, 1 GPU `#fef3c7`, 1 GPU/job `#fde8e6`. The right card is four independent jobs, not one job split across four GPUs.

```
Title at top in dark navy, centered: "One Flyte 2 pipeline."
Subtitle in gray: "Same workflow.py on a laptop, one GPU, or 1 GPU/job."

THREE equal machine cards in a row, rounded corners, plenty of padding. Little body text.

Left card, light blue wash (#dbeafe). Icon: MacBook laptop. Header: "MacBook Pro". Sub: "M4 Max, 128 GB, 16-core (12P+4E), MPS". Command pill: flyte run --local. One line: "full + LoRA".

Middle card, light yellow wash (#fef3c7). Icon: a single GPU card. Header: "1× RTX PRO 6000". Sub: "Blackwell, 96 GB, CUDA". Command pill: FLYTE_GPUS=1. One line: "serial 2×3 + QLoRA".

Right card, light pink wash (#fde8e6). Header: "4× RTX PRO 6000". Sub: "one node, 96 GB each, 1 GPU/job". Command pill: --parallel jobs. Visual: FOUR SEPARATE mini-job tiles in a 2×2 (GPU 0, GPU 1, GPU 2, GPU 3), each with one small GPU icon and the label "1 job". Do not draw four GPUs as one fused training job. One line: "four cells at once".

A downward chevron, then a short three-step row labeled workflow.py:
1 CPU: prepare_data
2 GPU: train / grid
3 CPU dashed: fetch_market_data (stub)

Do not name a university or cluster. Do not write DDP or 4 GPUs/model.
```

## `flyte_pipeline.png` (part 1 — text-to-SQL)

```
Title at top in dark navy, centered: "One Flyte 2 pipeline. Two machines."
Subtitle in gray: "Same workflow.py — flyte run --local here, flyte run on the GPU."

Two equal machine cards under the title, rounded corners, plenty of padding:

Left card, light blue wash, header "THIS MACBOOK PRO": Apple M4 Max, 128 GB, 16-core (12P+4E), MPS. Monospace pill: flyte run --local. Runs: SmolLM2-135M smoke, full + LoRA.

Right card, light green wash, header "REMOTE RTX PRO 6000 BLACKWELL": 96 GB each, CUDA. One GPU or four on one node. 16-core / 256 GB host. Monospace pill: flyte run. Runs: Union-scale samples, including QLoRA.

A downward chevron, then a horizontal four-step pipeline labeled "workflow.py":

Step 1 box, small gray CPU badge: prepare_data. Text: "SQL dataset on CPU. GPU stays free."

Arrow to Step 2 box, navy GPU badge: train. Text: "full / LoRA / QLoRA on SmolLM2."

Arrow to Step 3 box, navy GPU badge: evaluate. Text: "Exact-match SQL vs the base model."

Arrow to Step 4 box, dashed border, gray CPU badge: serve. Text: "Optional. Same run, not a new script."

Footer in small gray: "Swap --method or --model_name. Do not rewrite the pipeline for the next GPU."
```

## `hawkish_dovish_task.png` (part 2)

Left to right must match the paper ids: **0 Dovish, 1 Hawkish, 2 Neutral**. Do not put Neutral in the middle.

```
Title at top in dark navy, centered: "Hawkish–dovish classification"
Subtitle in gray: "Shah, Paturi, and Chava — Trillion Dollar Words (ACL 2023)"

THREE equal cards in a row, left to right in this exact order (class id, not a policy spectrum):

Left card, light green wash. Header: "DOVISH (0)". Sub: "easing / more accommodative". Quote: "The Committee decided to lower the target range for the federal funds rate."

Middle card, light red wash. Header: "HAWKISH (1)". Sub: "tightening / less accommodative". Quote: "The Committee judged that a further increase in the target range would be appropriate."

Right card, light gray wash. Header: "NEUTRAL (2)". Sub: "no clear stance". Quote: "Incoming data suggested that economic activity was expanding at a moderate pace."

Footer in small gray: "Labeled FOMC sentences, 1996–2022. Metric: weighted F1. Paper RoBERTa-large Combined-S: 0.711."

Example sentences are illustrative, not quoted from the paper.
```

## `ft_memory_budget.png` (part 2 — FOMC)

Unit-square GPU budget. Draw it in code (matplotlib) so LoRA yellow equals full yellow and purple is identical in all three columns. Do not copy a vendor LLaMA-65B / 780 GB slide.

```
Title: Where GPU memory goes
Subtitle: Four buckets. Trainable percent only shrinks grads and AdamW.

THREE equal cards, each a dashed 96 GB GPU outline. Stack bottom to top: yellow Parameters, blue Gradients, green AdamW (m, v), purple Activations.

FULL: Parameters 2×4, Gradients 2×4, AdamW 4×4, Activations 2×4. Caption: highest static memory.
LoRA: Parameters 2×4 (same as full), Gradients 1 square, AdamW 1 square, Activations 2×4 (same as full). Caption: same parameters and activations, tiny optimizer.
QLoRA: Parameters 2 squares (~4-bit vs bf16), Gradients 1, AdamW 1, Activations 2×4. Caption: lowest weight storage. CUDA only.

Legend + footer: Each square is one memory unit. LoRA keeps the full-size base in memory.
```

