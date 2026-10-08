# Part 2 — RoBERTa on FOMC hawkish–dovish

The Federal Open Market Committee sets the federal funds rate. That rate shows up in mortgages, car loans, credit cards, and savings yields. **Hawkish** language leans toward tighter policy (higher rates, cooler inflation, more expensive borrowing). **Dovish** language leans toward easier policy (lower rates, cheaper loans, more growth). [Shah et al. (2023)](https://aclanthology.org/2023.acl-long.368/) label FOMC sentences so you can measure that lean in text.

Full, LoRA, and QLoRA on Combined-S, same three seeds, weighted F1 vs their Table 5. **Nemotron-3-Nano-4B** is three more rows on that table (CUDA), not a second notebook. Flyte 2 is the orchestrator: one `workflow.py` on this MacBook Pro (`flyte run --local`), then the same tasks on an NVIDIA GPU. A training script is enough for one box; Flyte is what keeps CPU data tasks and GPU training as one DAG.

Labels:

| Id | Stance | Meaning |
|----|--------|---------|
| 0 | Dovish | Easing / more accommodative policy |
| 1 | Hawkish | Tightening / less accommodative policy |
| 2 | Neutral | No clear stance |

## What the paper trained

They fine-tuned both **RoBERTa-base** (~125M) and **RoBERTa-large** (~355M) on NVIDIA RTX A6000 GPUs. Base was the best of their small models (combined-split weighted F1 **0.698**). Large was best overall (**0.717** combined, **0.711** combined-split) and is the released checkpoint [`gtfintechlab/FOMC-RoBERTa`](https://huggingface.co/gtfintechlab/FOMC-RoBERTa). Data and code: [gtfintechlab/fomc-hawkish-dovish](https://github.com/gtfintechlab/fomc-hawkish-dovish) (CC BY-NC 4.0).

Their loop (see [`bert_fine_tune_lm_hawkish_dovish_train_test.py`](https://github.com/gtfintechlab/fomc-hawkish-dovish/blob/main/code_model/bert_fine_tune_lm_hawkish_dovish_train_test.py)): AdamW, max length 256, 20% of the official train file as validation, constant learning rate, early-stop patience 7 (reset if val CE **or** accuracy **or** F1 improves by 0.01), test the **last** weights. Combined-S grid winners: **RoBERTa-base lr 1e-5 batch 8**, **RoBERTa-large lr 1e-5 batch 16**.

`method=full` is that script. LoRA / QLoRA use the **same loop, seeds, LR, and batch**; adapters (or 4-bit) are the variation.

The labeled set is only ~2,400 sentences, so this is not a huge training job even at large size.

**Fair comparison:** our `full` / `LoRA` 3-seed mean ± std vs the paper’s Table 5 (same three seeds). The released Hugging Face checkpoint is a single set of weights, not a matched 3-seed re-run, so it is not in the mean table.

## Hardware: this MacBook Pro, then a remote NVIDIA RTX PRO 6000 Blackwell

| Machine | Specs | What to run |
|---------|-------|-------------|
| This laptop | 16-inch **MacBook Pro**, Apple **M4 Max**, **128 GB** unified memory, 16-core CPU (12P + 4E), 40-core GPU, PyTorch MPS | `roberta-base` and `roberta-large`, full + LoRA |
| Remote server | **1× or 4× NVIDIA RTX PRO 6000 Blackwell**, **96 GB** GDDR7 ECC each, one node (16-core, 256 GB host), CUDA | RoBERTa 2×3, then Nemotron-3-Nano-4B full / LoRA / QLoRA on the same table |
| GPU laptop | **Dell Pro Max 16 Plus**, **NVIDIA RTX PRO 5000 Blackwell** laptop GPU (**24 GB**), WSL2, CUDA | RoBERTa 2×3, Nemotron-3-Nano-4B **QLoRA** (~5.4 GB peak, ~1 h/seed with fused Mamba kernels), vLLM serving |
| Paper | RTX A6000 | Their original full fine-tunes (no LoRA/QLoRA) |

QLoRA needs **NVIDIA CUDA** because it uses [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes) 4-bit (NF4) quantization. That library does not provide a supported 4-bit training path on Apple MPS or CPU. `requirements.txt` therefore installs `bitsandbytes` only when `sys_platform != "darwin"`. On this Mac, `recommend_jobs()` omits QLoRA and `method="qlora"` raises a clear error — use `full` or `lora`. On the remote **RTX PRO 6000 Blackwell (96 GB)** run the **full 2×3**: RoBERTa-base and RoBERTa-large, each with full, LoRA, and QLoRA — not QLoRA-only.

## Comparison grid

Same data (Combined-S, seeds `5768`, `78516`, `944601`), same metric: **weighted F1**, reported as a **3-seed mean ± std** like Table 5.

| Model | Params | Full | LoRA | QLoRA |
|-------|--------|------|------|-------|
| `roberta-base` | ~125M | Mac follow-along; again on CUDA | Mac follow-along; again on CUDA | RTX PRO 6000 Blackwell only |
| `roberta-large` | ~355M | Mac follow-along; again on CUDA | Mac follow-along; again on CUDA | RTX PRO 6000 Blackwell only |
| `nemotron-nano-4b` | ~4B | CUDA (same Combined-S table) | CUDA | CUDA |

LoRA targets for RoBERTa are `query`, `key`, `value`, `dense` — not LLaMA `q_proj` names. The classification head is fully trained (`modules_to_save=["classifier"]`).

`delta_vs_paper` is our 3-seed mean minus their published 3-seed mean for that size. LoRA rows still use that full-FT number as the quality target.

## Results

The comparison table lives in the [notebook](fomc_roberta_finetune.ipynb) (**Results**). MacBook Pro (M4 Max, MPS) numbers are filled; CUDA RoBERTa and Nemotron rows fill as `metrics.json` lands. CUDA F1 is the **1 GPU** (yellow) block. `bash run_fomc.sh jobs` is the same recipe on four cards at once — same F1, not a second table block. We do not split one RoBERTa run across four cards. Re-run the results cell anytime.

Shah et al. Combined-S (Table 5, 3-seed mean ± std): RoBERTa-base **0.698 ± 0.010**, RoBERTa-large **0.711 ± 0.011**.

## Checkpoints and progress

Each cell writes:

```
checkpoints/<model>__<method>__<split>__seed5768/
  checkpoint-*       # Hugging Face Trainer (resume here)
  final/             # best weights / adapters after early stop
  metrics.json       # test scores; presence means “done”
  progress.jsonl     # one JSON object per step / eval / save
outputs/grid_runs.csv      # one row per seed
outputs/grid_summary.csv   # 3-seed mean ± std
```

Re-running **resumes** an interrupted job and **skips** a finished `metrics.json`. Pass `--force` (or `FORCE = True` in the notebook) to start over.

```bash
tail -f checkpoints/roberta-base__lora__lab-manual-split-combine__seed5768/progress.jsonl
```

Load a finished run:

```python
from train_run import load_trained, predict_texts

model, tokenizer = load_trained("checkpoints/roberta-base__lora__lab-manual-split-combine__seed5768")
predict_texts(model, tokenizer, ["The Committee decided to lower the target range."])
```

## Data (downloaded, not committed)

Default seed `5768`, Combined-S (`lab-manual-split-combine-*`):

```
https://raw.githubusercontent.com/gtfintechlab/fomc-hawkish-dovish/main/training_data/test-and-training/training_data/lab-manual-split-combine-train-5768.xlsx
https://raw.githubusercontent.com/gtfintechlab/fomc-hawkish-dovish/main/training_data/test-and-training/test_data/lab-manual-split-combine-test-5768.xlsx
```

`load_data.py` writes them under `data/`, which is gitignored. Use `split="lab-manual-combine"` if you want the non-split Combined numbers (paper large F1 0.717) instead.

## Setup

```bash
cd NLP/Fine_Tuning/02_fomc_roberta
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
python load_data.py
```

Or just:

```bash
cd NLP/Fine_Tuning/02_fomc_roberta
cp env.example .env         # optional; fill in local HF cache paths
bash run_fomc.sh            # in-process on this GPU node (`--local`, no Docker)
bash run_fomc.sh grid
bash run_fomc.sh grid-jobs
```

Notebook: [fomc_roberta_finetune.ipynb](fomc_roberta_finetune.ipynb).

## Flyte 2

Same **devbox** UI locally and on the remote GPU: **Sub actions** (child tasks), **Input** (typed args), **Output** (return values), plus Logs / Metrics / **Reports**. `prepare_data` is cached; a failed GPU cell retries without replaying the grid; the task image is `requirements.txt`.

| File | Role |
|------|------|
| `flyte_env.py` | CPU vs GPU task environments (`gpu=0` on this Mac so `--local` uses MPS) |
| `workflow.py` | Driver tasks (`main`, `pipeline`, `grid`) call leaf tasks (`prepare_data`, `train_one`, `snapshot_grid`, `write_talk_report`) |
| `paper_loop.py` | Shah et al. training loop (imported inside the GPU task, not at workflow top) |
| `config.yaml.example` | Dummy cluster config; copy to `.flyte/config.yaml` locally |

```bash
# This MacBook Pro (M4 Max) — CPU devbox. Do not pass --gpu (Apple Silicon unsupported).
flyte start devbox
flyte get devbox
flyte -c .flyte/devbox.yaml run --name fomc-main workflow.py main
# http://localhost:30080/v2  → run → Sub actions / Input / Output / Reports

# --local --tui is the terminal only. Those runs never appear at :30080.
flyte run --local --tui workflow.py pipeline --model_key roberta-base --method lora --seed 5768
flyte start tui
flyte run --local --tui workflow.py grid

# Remote NVIDIA RTX PRO 6000 Blackwell (96 GB) — GPU devbox, then the same flyte run
# (no --local). Full 2×3: (base + large) × (full + LoRA + QLoRA).
flyte start devbox --gpu
flyte get devbox
FLYTE_GPUS=1 flyte run workflow.py grid                  # one GPU, cells in series
FLYTE_GPUS=1 flyte run workflow.py grid --parallel jobs  # 1 GPU/job on a 4-GPU node
```

`bash run_fomc.sh jobs` is **1 GPU/job**: unfinished cells in parallel, one card each. F1 matches the 1-GPU row. Do not put one RoBERTa cell on four cards — the paper loop is single-device and the model already fits.

`python run_grid.py` launches those Flyte local runs per unfinished cell. `--no-flyte` is the in-process loop only.

## Serving with vLLM

QLoRA saves memory in training. Serving is a different stack: fold the adapter into the BF16 Nemotron base and serve it with [vLLM](https://github.com/vllm-project/vllm). The classifier scores three label tokens at the last prompt token, so vLLM generates one token restricted to those ids (`allowed_token_ids`). Same prompt, same argmax, same prediction.

vLLM pins its own torch, so it gets its own venv:

```bash
# --managed-python: vLLM compiles small C helpers and needs Python.h.
# flashinfer-jit-cache: prebuilt sampling kernels, so nothing needs nvcc at startup.
uv venv .venv-vllm --python 3.12 --managed-python
VIRTUAL_ENV=.venv-vllm uv pip install vllm peft pandas openpyxl scikit-learn
VIRTUAL_ENV=.venv-vllm uv pip install "flashinfer-jit-cache==$(.venv-vllm/bin/python -c 'import flashinfer; print(flashinfer.__version__)')" \
  --extra-index-url https://flashinfer.ai/whl/cu130

.venv/bin/python serve_vllm.py merge --method qlora --seed 5768          # -> checkpoints/<run>/merged
.venv-vllm/bin/python serve_vllm.py eval --method qlora --seed 5768      # test F1 + sentences/s
.venv-vllm/bin/python serve_vllm.py eval --method qlora --seed 5768 --lora   # unmerged adapter, hot-swappable
.venv-vllm/bin/vllm serve checkpoints/<run>/merged --served-model-name fomc-nemotron --max-model-len 512
```

`eval` writes `vllm_metrics.json` next to `metrics.json`, with the PyTorch F1 beside the vLLM F1.

On Flyte, `merge_adapter` (`workflow.py`) is a GPU task and `vllm_app` (`flyte_env.py`) is a [`VLLMAppEnvironment`](https://www.union.ai/docs/v2/flyte/user-guide/apps/native-app-integrations/vllm-app/) that serves that task's output behind an OpenAI-compatible API:

```bash
flyte run workflow.py merge_adapter --method qlora --seed 5768
flyte deploy workflow.py vllm_app
```

## CLI

```bash
# Recommended jobs × paper's 3 seeds (Flyte --local; skips finished metrics.json)
python run_grid.py

# One seed only
python run_grid.py --seed 5768
python train_run.py --model roberta-base --method lora --seed 78516

# One cell, no orchestrator
python train_run.py --model roberta-base --method lora
python train_run.py --model roberta-base --method full

# Remote RTX PRO 6000 Blackwell (96 GB): full 2×3 (base + large × full / LoRA / QLoRA)
python run_grid.py --include-qlora
flyte run workflow.py grid
```

## License

The FOMC annotations are CC BY-NC 4.0. Do not use them commercially. Cite the paper if you use the data or the released model:

```
@inproceedings{shah-etal-2023-trillion,
    title = "Trillion Dollar Words: A New Financial Dataset, Task {\&} Market Analysis",
    author = "Shah, Agam and Paturi, Suvan and Chava, Sudheer",
    booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    year = "2023",
    pages = "6664--6679",
    url = "https://aclanthology.org/2023.acl-long.368",
}
```
