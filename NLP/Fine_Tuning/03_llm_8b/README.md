# Part 3 — An ~8B LLM: full, LoRA, and QLoRA

Parts 1 and 2 use small models, so full fine-tuning already fits in memory and QLoRA looks like a no-op. An **8B** causal LM is where the three methods actually feel different — and where one **RTX PRO 6000 Blackwell (96 GB)** is enough to run **all three**, including full fine-tuning.

Same task as part 1: text-to-SQL on [`b-mc2/sql-create-context`](https://huggingface.co/datasets/b-mc2/sql-create-context). Same three methods. Bigger model.

**Default model:** [`meta-llama/Llama-3.1-8B`](https://huggingface.co/meta-llama/Llama-3.1-8B) (Meta, 8B). Gated on Hugging Face — accept the license and set `HF_TOKEN` (see part 1 `.env.example`).

**No-gate fallback:** [`mistralai/Mistral-7B-v0.3`](https://huggingface.co/mistralai/Mistral-7B-v0.3) (~7B). Same LoRA module names (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`).

We use the **base** checkpoints, not Instruct, so the before/after SQL eval is as clear as it was for SmolLM2-135M.

## Why 8B is the interesting size

Rough bf16 footprint for an 8B model:

| Piece | Size |
|-------|------|
| Weights | ~16 GB |
| Gradients | ~16 GB |
| AdamW states (fp32 m + v) | ~64 GB |
| **Subtotal before activations** | **~96 GB** |

So vanilla full fine-tuning of 8B is the line that a workstation GPU can cross and a laptop cannot. LoRA only trains adapters (a few tens of millions of params). QLoRA stores the frozen base in 4-bit (~4–6 GB) and is the path for a 16–24 GB card.

| Method | What you need | On a Mac | On NVIDIA RTX PRO 6000 Blackwell (96 GB) |
|--------|----------------|----------|-------------------------|
| Full | ~96 GB + activations; use gradient checkpointing and a small batch | No | Yes — one card has headroom past the ~96 GB optimizer footprint |
| LoRA | ~20–24 GB | Uncomfortable / slow if at all | Easy |
| QLoRA | ~6–12 GB | Experimental (bitsandbytes MPS) | Easy; also copyable by readers on a smaller NVIDIA GPU |

Hardware for the reported numbers is the same as part 2: **NVIDIA RTX PRO 6000 Blackwell (96 GB)**, one card or four on one node (16-core, 256 GB host).

## Other models that drop in

All of these use LLaMA-style `*_proj` names, so the part 1 Flyte pipeline can swap the string:

| Model | Params | Notes |
|-------|--------|--------|
| `meta-llama/Llama-3.1-8B` | 8B | Default. Needs a Hugging Face license + token |
| `mistralai/Mistral-7B-v0.3` | 7B | No gate; closest ungated substitute |
| `google/gemma-2-9b` | 9B | Slightly larger; check `target_modules` if LoRA reports 0 params |
| `allenai/OLMo-2-1124-7B` | 7B | Fully open weights |

Stay on this list unless you have a reason to change architecture. If LoRA trains 0 parameters, print module names (see part 1 README).

## How to run (reuse part 1)

The Union/Flyte 2 pipeline already takes `--model_name` and `--method`. From `01_text_to_sql/`:

```bash
# LoRA (start here)
flyte run --local --tui workflow.py pipeline \
  --model_name meta-llama/Llama-3.1-8B \
  --method lora --epochs 1 \
  --max-train-samples 2000 --max-eval-samples 200 --batch_size 1

# QLoRA
flyte run --local --tui workflow.py pipeline \
  --model_name meta-llama/Llama-3.1-8B \
  --method qlora --epochs 1 \
  --max-train-samples 2000 --max-eval-samples 200 --batch_size 1

# Full fine-tune — RTX PRO 6000 Blackwell (96 GB), not a Mac
flyte run --local --tui workflow.py pipeline \
  --model_name meta-llama/Llama-3.1-8B \
  --method full --epochs 1 \
  --max-train-samples 2000 --max-eval-samples 200 --batch_size 1
```

Ungated:

```bash
--model_name mistralai/Mistral-7B-v0.3
```

Full 8B fine-tuning may need **gradient checkpointing** (and possibly 8-bit Adam) so activations fit next to the ~96 GB of weights + optimizer. That tweak lives in `workflow.py` when we wire this part to actually run; LoRA/QLoRA should work with the current script.

Start with a short subset (`--max-train-samples 100`) to prove download + train + eval before a longer run.

## Compare

Same eval as part 1 (exact-match SQL, base vs fine-tuned) plus a resource table:

| Model | Method | Trainable params | Peak VRAM | Wall time | SQL exact match |
|-------|--------|------------------|-----------|-----------|-----------------|
| SmolLM2-135M (part 1) | full / lora / qlora | … | … | … | … |
| Llama-3.1-8B | full / lora / qlora | … | … | … | … |

The story we want in the write-up: 8B full fine-tuning is doable on one modern workstation GPU; LoRA gets most of the quality for a fraction of the trainable weights; QLoRA is how you do the same adaptation on a much smaller card.
