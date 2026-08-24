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

```
Title at top in dark navy, centered: "One Flyte 2 pipeline."
Subtitle in gray: "Same workflow.py on a laptop, one GPU, or four GPUs."

THREE equal machine cards in a row, rounded corners, plenty of padding. Little body text.

Left card, light blue wash. Icon: MacBook laptop. Header: "MacBook Pro". Sub: "M4 Max, 128 GB, 16-core (12P+4E), MPS". Command pill: flyte run --local. One line: "full + LoRA".

Middle card, light green wash. Icon: a single GPU card. Header: "1× RTX PRO 6000". Sub: "Blackwell, 96 GB, CUDA". Command pill: FLYTE_GPUS=1. One line: "2×3 + QLoRA".

Right card, slightly darker green wash. Icon: four GPU cards in a row. Header: "4× RTX PRO 6000". Sub: "one node, 96 GB each, 16-core / 256 GB host". Command pill: --parallel jobs. One line: "1 GPU/job".

A downward chevron, then a short three-step row labeled workflow.py:
1 CPU: prepare_data
2 GPU: train / grid
3 CPU dashed: fetch_market_data (stub)

Do not name a university or cluster.
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
