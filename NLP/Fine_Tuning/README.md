# Fine-Tuning with LoRA and QLoRA

Code and slides behind the ODSC AI West talk and the ODSC blog post [LoRA and QLoRA fine-tuning with Flyte and vLLM](https://opendatascience.com/lora-qlora-fine-tuning-flyte-vllm/).

| Folder | What it is |
|--------|------------|
| [fomc_peft](fomc_peft/) | Hawkish/dovish classification on FOMC sentences: **RoBERTa-base / large**, then **Nemotron 3 Nano 4B** on the same Combined-S table. Full / LoRA / QLoRA. A **Flyte 2** pipeline (`workflow.py`) runs it on a laptop with `flyte run --local` or on a remote NVIDIA GPU with `flyte run`. |
| [fomc_peft/slides](fomc_peft/slides/) | The talk deck (`presentation_peft_talk.pptx`) and a PDF export. |

## Why this lives under `NLP/`

This repo's public tutorials already point at paths like `Pandas/`, `Sklearn/`, `RAG/`. Those folders stay put. NLP was an empty section in the root README, so a new `NLP/` tree is additive and does not break old links.

`RAG/` stays at the repo root (same reason). Fine-tuning is NLP; retrieval is a separate tutorial.

## Methods we compare

| Method | What trains | Typical memory | When it is useful |
|--------|-------------|----------------|-------------------|
| Full | Every weight | Highest | Small models, closest match to a paper's reported numbers |
| LoRA | Low-rank adapters only | Medium | Most of the quality, far fewer trainable params |
| QLoRA | LoRA adapters + 4-bit frozen base | Lowest | Models that would not otherwise fit in VRAM |

QLoRA is [Dettmers et al., 2023](https://arxiv.org/abs/2305.14314). It needs **NVIDIA CUDA**: 4-bit quantization goes through [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes), which we do not install on macOS (`bitsandbytes>=0.44.0; sys_platform != "darwin"`). On a Mac or CPU, use full or LoRA. The FOMC paper is a different arXiv id ([2305.07972](https://arxiv.org/abs/2305.07972)).

On RoBERTa-base, QLoRA is pedagogical: the model already fits in memory. RoBERTa-large starts to show a memory gap. **Nemotron-3-Nano-4B** is the first causal LM on that same FOMC table: full fine-tune fits on a 96 GB Blackwell; LoRA and QLoRA are the cheaper rows.

Follow along on this **16-inch MacBook Pro** (Apple **M4 Max**, **128 GB** unified memory, MPS) for RoBERTa full + LoRA (`flyte run --local`). On a remote **NVIDIA RTX PRO 6000 Blackwell (96 GB)** or H200, run the FOMC 2×3 plus the Nemotron rows: same `workflow.py`, GPU job instead of the laptop.

## What is gitignored

Checkpoints, model weights, Hugging Face caches, Flyte/Union config, `.env` files, and the downloaded FOMC xlsx files are ignored. Tutorial sample CSVs and the other small datasets already in this repo stay. See `NLP/.gitignore` and the repo-root `.gitignore`.
