# Code guide: FOMC fine-tuning (RoBERTa vs Nemotron, full / LoRA / QLoRA)

A map of this folder: what each file does, which code produced each number and
image in the ODSC blog, and how to rerun it. The `README.md` is the tutorial;
this file is for inspecting the code.

Everything ran on the Dell laptop GPU (RTX PRO 5000 Blackwell, 24 GB, WSL2).

## Folder layout

```
02_fomc_roberta/
├── fomc_roberta_finetune.ipynb   notebook (tutorial walkthrough)
├── workflow.py                   Flyte 2 pipeline (tasks, reports, release gate)
├── flyte_env.py                  Flyte task environments: CPU, GPU, UI images
├── train_run.py                  one run: full / LoRA / QLoRA / zero-shot
├── paper_loop.py                 Shah et al. training loop (ported)
├── causal_cls.py                 Nemotron as a classifier (label-token logits)
├── fomc_prompt.py                prompt that wraps a sentence for Nemotron
├── load_data.py                  downloads the paper's train/test splits
├── config.py                     models, seeds, hyperparameters, run paths
├── device_utils.py               CUDA / MPS / CPU detection, memory helpers
├── run_grid.py                   runs many cells, writes outputs/grid_*.csv
├── run_parallel.py               one process per GPU for unfinished cells
├── run_fomc.sh                   shell entry point (sets up Mamba kernels)
├── build_mamba_kernels.sh        builds causal-conv1d (fast Nemotron training)
├── patch_mamba_ssm.py            makes mamba-ssm importable without its CUDA build
├── report_stages.py              HTML report per pipeline stage (data, train, compare, release)
├── report_viz.py                 inline-SVG charts used by the reports
├── report_helpers.py             loads results, writes report_snapshot.py
├── report_snapshot.py            auto-generated: all finished runs, bundled for Flyte
├── serve_vllm.py                 merge adapter + score the test set with vLLM
├── bench_inference.py            inference benchmark (F1, sentences/s, latency, memory)
├── paper_baselines.py            the paper's published numbers
├── devbox/wsl-gpu-podtemplate.yaml   lets the local Flyte devbox use the WSL GPU
├── images/                       figures used by the notebook
├── checkpoints/   (49 GB, not in git)   one folder per run: metrics.json, predictions, adapters
├── outputs/       grid_summary.csv, inference_bench.jsonl, report_snapshot.json
└── logs/          training and benchmark logs
```

## How a run works

```
load_data.py ──► train_run.py ──► paper_loop.py ──► checkpoints/<run>/metrics.json
                     │                                   test_predictions.jsonl
                     ├── RoBERTa: sequence-classification head on the raw sentence
                     └── Nemotron: causal_cls.LabelTokenClassifier
                           fomc_prompt.py wraps the sentence; the model is scored on the
                           LM-head logits of the three label words (dovish/hawkish/neutral)
```

- **full**: every weight trained (RoBERTa only; Nemotron does not fit in 24 GB).
- **lora**: frozen BF16 base + rank-64 adapters (`config.py: NEMOTRON_LORA_*`).
- **qlora**: same adapters, base stored in 4-bit NF4 (`train_run.py: _replace_linears_with_nf4`).
  Kept in BF16: `lm_head` (scores the labels) and Mamba `out_proj` (the fused kernel reads it).
- **zeroshot**: no training. RoBERTa = untrained head; Nemotron = prompt only
  (`train_run.py: run_zero_shot`).

Run folders are named `checkpoints/<model>__<method>__lab-manual-split-combine__seed<seed>__<recipe>`.

## Where each blog number and image comes from

| Blog item | Produced by | Data |
|---|---|---|
| Results for all 33 runs (3 seeds each) | `train_run.py` (Nemotron), `run_grid.py` (RoBERTa) | `checkpoints/*/metrics.json` |
| Nemotron QLoRA 0.716, LoRA 0.727, zero-shot 0.622 | same | `outputs/report_snapshot.json` |
| Peak training memory (5.4 GB QLoRA, 9.0 GB LoRA) | `device_utils.peak_memory_gb` during training | `metrics.json` → `peak_mem_gb` |
| Flyte screenshots "Why fine-tune? (QLoRA)" / "(LoRA)" | `workflow.py: main` → `report_stages.grid_report` → `why_finetune_section` | devbox run `fomc-stage3-why-two` |
| FP8 serving: 0.725, ~100 sentences/s, 5.2 GB | `bench_inference.py nemotron --method lora --variant lora-fp8` | `outputs/inference_bench.jsonl` |
| QLoRA code snippet | written for the blog; equivalent to `train_run.py` (which swaps in `bnb.nn.LinearNF4` by hand) | n/a |

The screenshots themselves are captured by
`pptx-audio-system/decks/peft_talk/build_scripts/flyte_ui/capture_stage3_report.js`, and the
hero memory diagram is built by `build_scripts/build_blog_hero_memory.py` in the same repo.

## Rerunning

```bash
source .venv/bin/activate                    # training venv
bash run_fomc.sh nemotron-lora               # one-time: Mamba kernels + one Nemotron LoRA run

# Nemotron, one seed (about an hour each)
python train_run.py --model nemotron-nano-4b --method qlora --seed 5768
python train_run.py --model nemotron-nano-4b --method lora  --seed 5768
python train_run.py --model nemotron-nano-4b --method zeroshot --seed 5768

# RoBERTa grid
python run_grid.py --models roberta-base,roberta-large --methods full,lora,qlora \
    --seeds 5768,78516,944601 --include-qlora --include-large-full

# Refresh the results bundled into Flyte reports
python -c "from report_helpers import refresh_report_snapshot; refresh_report_snapshot()"

# Flyte: comparison report on the local devbox (UI at http://localhost:30080/v2)
FLYTE_WSL_GPU=1 flyte -c .flyte/devbox.yaml run --copy-style all workflow.py main

# Serving (separate venv; vLLM pins its own torch)
.venv/bin/python serve_vllm.py merge --method lora --seed 5768
.venv-vllm/bin/python bench_inference.py nemotron --method lora --variant lora-fp8 --seed 5768
.venv/bin/python bench_inference.py roberta --model roberta-large --method full --seed 5768
```

Flyte bundles only git-tracked files, so `git add` new files before a devbox run.

## Not in git

`checkpoints/` (49 GB of run folders and adapters), `outputs/`, `logs/`, `data/`, and the two
virtual environments stay local. Rerun the commands above to regenerate them.
