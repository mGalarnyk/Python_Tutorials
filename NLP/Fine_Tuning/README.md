# Fine-Tuning with LoRA and QLoRA

Part 1 and part 2 are **Flyte 2** pipelines so the same `workflow.py` is a laptop command (`flyte run --local`) and a remote NVIDIA GPU command (`flyte run`): CPU tasks for data, GPU tasks for training. That is the point — not a Flyte 1→2 SDK footnote. Part 2 also keeps Shah et al.’s loop so 3-seed F1 can sit next to Table 5, and leaves a CPU hook for later market data.

| Folder | Status | What it is |
|--------|--------|------------|
| [01_text_to_sql](01_text_to_sql/) | Ready to run | Text-to-SQL on SmolLM2-135M. Adapted from [Union AI's workshop](https://github.com/unionai/workshops/tree/main/tutorials/llm-fine-tuning-lora-qlora). Already **Flyte 2**. |
| [02_fomc_roberta](02_fomc_roberta/) | Ready to run | Classification: **RoBERTa-base and RoBERTa-large**, full / LoRA / QLoRA vs Shah et al. Combined-S F1. **Flyte 2**. Checkpoints resume. |
| [03_llm_8b](03_llm_8b/) | Scaffold | Same SQL task as part 1 on **Llama-3.1-8B** (fallback: Mistral-7B). Full / LoRA / QLoRA; full FT is the remote CUDA run. |

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

On SmolLM2-135M and RoBERTa-base, QLoRA is pedagogical — both already fit in memory. RoBERTa-large starts to show a memory gap. **Llama-3.1-8B** is where full fine-tuning needs a workstation (~96 GB for weights + AdamW before activations), LoRA fits in ~24 GB, and QLoRA fits in ~6–12 GB.

Follow along on this **16-inch MacBook Pro** (Apple **M4 Max**, **128 GB** unified memory, 16-core CPU (12P + 4E), 40-core GPU, MPS) for part 1 and for RoBERTa full + LoRA. On the remote **NVIDIA RTX PRO 6000 Blackwell (96 GB)** — one card or **four on one node** (16-core, 256 GB host) — run the FOMC **2×3** (base and large, each of full / LoRA / QLoRA) and the ~8B full fine-tune (CUDA / bitsandbytes).

## What is gitignored

Checkpoints, Hugging Face caches, Flyte/Union config (cluster URLs), and `.env` files are ignored so local GPU runs do not get committed. See `NLP/.gitignore` and the repo-root `.gitignore`.
