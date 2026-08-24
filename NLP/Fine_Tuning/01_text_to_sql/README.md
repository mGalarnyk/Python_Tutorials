# Part 1 — Text-to-SQL (full / LoRA / QLoRA)

Fine-tune a small causal LM on text-to-SQL with **full fine-tuning**, **LoRA**, or **QLoRA**. Adapted from Union AI's [llm-fine-tuning-lora-qlora](https://github.com/unionai/workshops/tree/main/tutorials/llm-fine-tuning-lora-qlora) workshop. Default model: [SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M).

**Why Flyte 2:** a one-off `Trainer` script is enough for a single laptop run. This tutorial is two machines and more than one kind of work. Follow along on this **MacBook Pro** (M4 Max, 128 GB, 16-core 12P+4E, MPS) with a smoke test. Union-scale samples and QLoRA run on a remote **NVIDIA RTX PRO 6000 Blackwell (96 GB)**. Part 3 swaps in an 8B model on the same DAG. Without Flyte that is three scripts. With it, one `workflow.py`: CPU `prepare_data`, GPU `train` / `evaluate`, optional serve. `flyte run --local` here; `flyte run` on the GPU. Change `--method` or `--model_name`, not the pipeline.

## SDK note (already Flyte 2)

This workshop does **not** need a Flyte 1 → 2 migration. The upstream code already uses the Flyte 2 SDK:

| Flyte 1 | This tutorial (Flyte 2) |
|---------|-------------------------|
| `import flytekit` | `import flyte` |
| `@task` / `@workflow` | `TaskEnvironment` + `@cpu_env.task` / `@gpu_env.task` |
| `FlyteDirectory` | `flyte.io.Dir` |
| `pyflyte run` | `flyte run` |
| Decks | `report=True` |

`requirements.txt` pins `flyte[tui]>=2.0`. See Union's [Flyte 1 → 2 guide](https://www.union.ai/docs/v2/flyte/user-guide/migration/flyte-2/) if you later port other Flyte 1 code.

## What's here

| File | Role |
|------|------|
| `device_utils.py` / `run_profile.py` | CUDA vs MPS vs CPU; laptop vs GPU run sizes |
| `config.py` | Flyte environments — CPU for data prep, GPU for training (`gpu=0` on a Mac so `--local` works) |
| `workflow.py` | prepare data → train → evaluate base vs fine-tuned |
| `report_helpers.py` | HTML/SVG reports for the Flyte UI |
| `serve.py` | FastAPI serving (optional, needs a Flyte cluster) |
| `app_gradio.py` | Gradio UI in front of `serve.py` (optional) |
| `llm-fine-tune-tutorial.ipynb` | Notebook: pipeline diagram first, then train; placeholders, not a personal GPU endpoint |
| `images/flyte_pipeline.png` | High-level DAG: two machines, CPU vs GPU tasks |

## Setup

```bash
cd NLP/Fine_Tuning/01_text_to_sql

uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
```

Optional Hugging Face token for gated models (never commit `.env`):

```bash
cp .env.example .env
# then edit .env
```

Remote Flyte/Union config is **not** committed. If you use a cluster, generate it locally:

```bash
flyte create config \
  --endpoint YOUR_CLUSTER.hosted.unionai.cloud \
  --project YOUR_PROJECT \
  --domain development \
  --builder remote
```

That writes `.flyte/config.yaml`, which is gitignored. A dummy layout is in `config.yaml.example`.

## How LoRA and QLoRA differ from full fine-tuning

**Full fine-tuning** updates every weight. Effective, but expensive to train, store, and deploy.

**LoRA** freezes the base weights `W` and trains a low-rank update `A × B`, scaled by `alpha / r`:

```
                    ┌─────────────────────────┐
                    │   Original Weight W      │
input ────────────→ │   (frozen)               │──→ main output
    │               └─────────────────────────┘         │
    │               ┌───────────┐ ┌───────────┐         │
    └─────────────→ │ A (d × r)  │→│ B (r × d)  │→ × α/r ──→ + ──→ combined
                    └───────────┘ └───────────┘
                    (trainable adapters)
```

- `r` — adapter rank (capacity vs size)
- `alpha` — scale; common default is `alpha = 2 * r`. Adds no extra parameters.

**QLoRA** keeps those adapters in higher precision but stores the frozen base in 4-bit (NF4). That 4-bit path is [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes), which **requires NVIDIA CUDA** — not Apple MPS or CPU. `requirements.txt` therefore installs `bitsandbytes` only when `sys_platform != "darwin"`. On a Mac, use `lora` or `full`; `method=qlora` raises a clear error. Run QLoRA on the remote **NVIDIA RTX PRO 6000 Blackwell (96 GB)** (same pipeline).

On SmolLM2-135M, QLoRA is overkill (quality often drops) and is here so the same `method` flag works when you swap in a larger model.

LoRA targets in `workflow.py` are LLaMA-style names (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`). RoBERTa uses different names — that is handled in part 2.

## Run

Start small and **local**. The notebook (`llm-fine-tune-tutorial.ipynb`) picks MPS vs CUDA and a smoke-test size automatically.

```bash
# Mac laptop (MPS) or any machine without NVIDIA — LoRA smoke test
flyte run --local workflow.py pipeline \
  --method lora --epochs 1 --batch_size 2 \
  --max_train_samples 100 --max_eval_samples 20 --num_eval_examples 10

# Remote NVIDIA RTX PRO 6000 Blackwell (96 GB) — Union-scale LoRA
flyte run --local workflow.py pipeline --method lora --epochs 3

# QLoRA — NVIDIA CUDA only (bitsandbytes 4-bit; skipped on Mac)
flyte run --local workflow.py pipeline --method qlora
```

On a Mac, `recommend_jobs()` omits QLoRA. Treat CUDA as the QLoRA path.

On a Flyte 2 cluster you already configured (config file stays local):

```bash
FLYTE_GPUS=1 flyte run workflow.py pipeline --method lora --epochs 3
FLYTE_GPUS=4 flyte run workflow.py pipeline --method lora --epochs 3
```

`FLYTE_GPUS` is how many NVIDIA GPUs the train task requests (default 1). Same `workflow.py` for one GPU or four on one node. Keep `FLYTE_GPUS=1` unless a single task should own the node.

### Useful flags

| Flag | Default | Meaning |
|------|---------|---------|
| `--model_name` | `HuggingFaceTB/SmolLM2-135M` | Hugging Face model |
| `--dataset_name` | `b-mc2/sql-create-context` | Hugging Face dataset |
| `--method` | `lora` | `full`, `lora`, or `qlora` |
| `--epochs` | `3` | Training epochs |
| `--lr` | `2e-4` | Learning rate |
| `--batch_size` | `4` | Per-device batch size |
| `--max_train_samples` | `5000` | Cap on train size |
| `--max_eval_samples` | `500` | Cap on eval size |
| `--num_eval_examples` | `50` | Examples in the before/after report |
| `--lora_r` | `16` | LoRA rank |
| `--lora_alpha` | `32` | LoRA alpha |

## Evaluation

The evaluate task runs the same prompts on the base model and the fine-tuned model:

- Exact-match accuracy on normalized SQL
- Side-by-side raw outputs (base models often ramble; fine-tuned models usually stop after the query)

**Results live in the [notebook](llm-fine-tune-tutorial.ipynb) (section 3).** That table is full vs LoRA vs QLoRA (QLoRA only when CUDA/bitsandbytes is available). Re-run the results cell anytime — it reads `outputs/run_<method>.json`.

```bash
python run_grid.py              # smoke: full + LoRA (QLoRA if NVIDIA)
python run_grid.py --no-smoke   # Union-scale sample counts (RTX PRO 6000 Blackwell)
```

## Serve (optional)

Only after a successful training run, and only if you have a Flyte cluster:

```bash
python serve.py
# python serve.py --run-name YOUR_RUN_NAME
```

Point curl at **your** deployed URL, not a sample host:

```bash
curl -X POST https://YOUR_APP_URL/generate \
  -H "Content-Type: application/json" \
  -d '{
    "schema": "CREATE TABLE employees (id INT, name VARCHAR, department VARCHAR, salary INT)",
    "question": "What is the average salary by department?"
  }'
```

Gradio UI:

```bash
SERVER_URL=https://YOUR_APP_URL python app_gradio.py
```

## Next

[Part 2 — RoBERTa on FOMC hawkish–dovish](../02_fomc_roberta/) uses the same three methods on a classification task from [Trillion Dollar Words](https://arxiv.org/abs/2305.07972).
