# Fine-Tuning with LoRA and QLoRA

Part 1 and part 2 are **Flyte 2** pipelines so the same `workflow.py` is a laptop command (`flyte run --local`) and a remote NVIDIA GPU command (`flyte run`): CPU tasks for data, GPU tasks for training. That is the point — not a Flyte 1→2 SDK footnote. Part 2 also keeps Shah et al.’s loop so 3-seed F1 can sit next to Table 5, and leaves a CPU hook for later market data.

| Folder | Status | What it is |
|--------|--------|------------|
| [01_text_to_sql](01_text_to_sql/) | Ready to run | Text-to-SQL on SmolLM2-135M. Adapted from [Union AI's workshop](https://github.com/unionai/workshops/tree/main/tutorials/llm-fine-tuning-lora-qlora). Already **Flyte 2**. |
| [02_fomc_roberta](02_fomc_roberta/) | Ready to run | Classification: **RoBERTa-base / large**, then **Nemotron-3-Nano-4B** on the same Combined-S table. Full / LoRA / QLoRA. **Flyte 2**: laptop `--local`, then a remote NVIDIA GPU. |
| [03_llm_8b](03_llm_8b/) | Deferred | Was Llama-3.1-8B / SQL. The 4B model lives in the part 2 FOMC table instead. |

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

On SmolLM2-135M and RoBERTa-base, QLoRA is pedagogical — both already fit in memory. RoBERTa-large starts to show a memory gap. **Nemotron-3-Nano-4B** is the first causal LM on that same FOMC table: full fine-tune fits on a 96 GB Blackwell; LoRA and QLoRA are the cheaper rows.

Follow along on this **16-inch MacBook Pro** (Apple **M4 Max**, **128 GB** unified memory, MPS) for part 1 and for RoBERTa full + LoRA (`flyte run --local`). On a remote **NVIDIA RTX PRO 6000 Blackwell (96 GB)** or H200, run the FOMC 2×3 plus the Nemotron rows: same `workflow.py`, GPU job instead of the laptop.

## What is gitignored

Checkpoints, model weights, Hugging Face caches, Flyte/Union config, `.env` files, and the downloaded FOMC xlsx files are ignored. Tutorial sample CSVs and the other small datasets already in this repo stay. See `NLP/.gitignore` and the repo-root `.gitignore`.
