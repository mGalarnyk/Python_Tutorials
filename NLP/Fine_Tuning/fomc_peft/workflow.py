"""
FOMC hawkish-dovish Flyte 2 pipeline.

Each function below with an @<env>.task decorator is a Flyte task. When one task
calls another (pipeline -> prepare_data -> train_one), Flyte records each call
as its own step ("action") with its inputs, outputs, logs, and report, so a run
in the UI shows the whole tree. The environments (container image, CPU / GPU)
are defined in flyte_env.py.

The training itself is the same paper loop as run_grid.py / train_run.py;
Flyte adds caching, retries, per-step reports, parallel runs, and an approval
gate on top.

Usage:
    # In this Python process, no Docker:
    flyte run --local workflow.py train_one --model_key nemotron-nano-4b --method qlora --seed 5768
    flyte run --local workflow.py grid --seeds '[5768]' --include_qlora true --include_nemotron
    flyte run --local workflow.py release              # y: merge the adapter for vLLM
    # On a local Flyte devbox, with the UI at http://localhost:30080/v2:
    flyte start devbox
    FLYTE_WSL_GPU=1 flyte -c .flyte/devbox.yaml run --copy-style all workflow.py main
    # Serve the fine-tuned Nemotron with vLLM (OpenAI-compatible API):
    flyte run workflow.py merge_adapter --method lora --seed 5768
    flyte deploy workflow.py vllm_app
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import timedelta
from pathlib import Path
from dataclasses import dataclass

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import flyte
import flyte.report  # required — `import flyte` does not load this submodule
import flyte.report._report

# flyte 2.11 bug: under IPython, Report.get_final_report() returns an IPython HTML
# object, and flush() (ours and the task runner's) asserts str, so every
# report=True task fails in a notebook. Always render the report as a string.
flyte.report._report.ipython_check = lambda: False

import config as cfg
from flyte_env import cpu_env, gpu_env, ui_env, vllm_app  # noqa: F401 (vllm_app: flyte deploy)
from report_helpers import (
    TALK_SENTENCES,
    fomc_talk_report,
    load_completed_runs,
    load_talk_reads,
    market_data_report,
    persist_talk_reads,
    reads_from_predict,
    source_label,
    refresh_report_snapshot,
    summarize_grid,
    table5_tab_html,
)

logging.basicConfig(level=logging.WARNING, format="%(message)s", force=True)
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

# Do not import train_run / torch at module top (fork + MPS).


# Task outputs are typed. A dataclass return value shows up field by field in
# the Output tab of the Flyte UI, and other tasks can take it as an input.
@dataclass
class CellOut:
    """Named Summary → Output for one paper-loop cell."""

    model: str
    method: str
    seed: int
    weighted_f1: float
    vs_paper: float
    device: str
    status: str
    labels: str


@dataclass
class GridOut:
    """Named Summary → Output for the live grid."""

    cells_done: int
    mac: str
    cuda: int
    talk_source: str
    labels: str
    updated: str


def _label_csv(reads: list[dict] | None) -> str:
    if not reads:
        return ""
    return ", ".join(str(r.get("pred") or "") for r in reads)


def _grid_out(grid: dict, reads: list[dict] | None = None) -> GridOut:
    talks, src = load_talk_reads(reads)
    return GridOut(
        cells_done=int(grid["n_complete"]),
        mac=f"{grid['n_mac']}/{grid['n_mac_expected']}",
        cuda=int(grid["n_cuda"]),
        talk_source=str(grid.get("talk_source") or src or ""),
        labels=_label_csv(talks),
        updated=str(grid["updated"]),
    )


def _cell_out(result: dict, reads: list[dict] | None) -> CellOut:
    test = result.get("test") or {}
    f1 = test.get("weighted_f1")
    delta = result.get("delta_vs_paper")
    return CellOut(
        model=str(result.get("model") or ""),
        method=str(result.get("method") or ""),
        seed=int(result.get("seed") or 0),
        weighted_f1=float(f1) if f1 is not None else -1.0,
        vs_paper=float(delta) if delta is not None else 0.0,
        device=str(result.get("device") or ""),
        status=str(result.get("status") or ""),
        labels=_label_csv(reads),
    )

async def _publish_training_curve(
    progress: Path, model_key: str, method: str, seed: int, shown: int, result: dict | None = None
) -> int:
    """Redraw the Training tab when progress.jsonl has new epochs. shown=-1 forces a redraw.

    With the finished result it also adds the error analysis from test_predictions.jsonl.
    """
    from report_stages import load_predictions, training_report, zero_shot_report

    # Flyte reports: a task with report=True gets an HTML report in the UI.
    # get_tab("Training") adds a named tab; replace() sets its HTML; flush()
    # sends it to the UI now, so the chart updates while the task is still running.
    if result and result.get("method") == "zeroshot":
        html = zero_shot_report(model_key, seed, result, load_predictions(progress.parent))
        flyte.report.get_tab("Training").replace(html)
        await flyte.report.flush.aio()
        return shown
    if not progress.exists():
        return shown
    evals = [r for r in map(json.loads, progress.read_text().splitlines()) if r.get("event") == "eval"]
    if not evals or (shown >= 0 and len(evals) == shown):
        return shown
    preds = load_predictions(progress.parent) if result else None
    html = training_report(evals, model_key, method, seed, metrics=result, preds=preds)
    flyte.report.get_tab("Training").replace(html)
    await flyte.report.flush.aio()
    return len(evals)


async def _flush_talk_report(
    html: str,
    reads: list[dict] | None = None,
    grid: dict | None = None,
    highlight: tuple[str, str] | None = None,
) -> None:
    """Official Flyte 2 report API: default tab + extra tabs, then flush."""
    grid = grid or summarize_grid()
    await flyte.report.replace.aio(html)
    flyte.report.get_tab("Table 5").log(table5_tab_html(grid, highlight=highlight))
    await flyte.report.flush.aio()


def _say(msg: str) -> None:
    """Loud progress so a terminal / Flyte log shows each cell and model."""
    line = f"[fomc] {msg}"
    log.info(line)
    print(line, flush=True)


def _metrics_complete(model_key: str, method: str, seed: int, split: str) -> bool:
    path = cfg.run_dir(model_key, method, seed, split) / "metrics.json"
    if not path.exists():
        return False
    try:
        return json.loads(path.read_text()).get("status") == "complete"
    except json.JSONDecodeError:
        return False


def _announce_cell(cell: CellOut, index: int, total: int, split: str, jobs: list[tuple[str, str]]) -> None:
    f1 = cell.weighted_f1
    f1_s = f"{f1:.4f}" if f1 is not None and f1 >= 0 else "—"
    _say(
        f"DONE cell {index}/{total}: {cell.model} {cell.method} seed={cell.seed}  "
        f"F1={f1_s} vs_paper={cell.vs_paper:+.4f}  {cell.status}"
    )
    n_seeds = len(cfg.PAPER_SEEDS)
    seeds_done = sum(
        _metrics_complete(cell.model, cell.method, int(s), split) for s in cfg.PAPER_SEEDS
    )
    if seeds_done == n_seeds:
        _say(f"DONE method: {cell.model} {cell.method}  (all {n_seeds} seeds)")
    methods = [m for mk, m in jobs if mk == cell.model]
    if methods and all(
        all(_metrics_complete(cell.model, m, int(s), split) for s in cfg.PAPER_SEEDS)
        for m in methods
    ):
        _say(f"DONE model: {cell.model}  ({len(methods)} methods × {n_seeds} seeds)")


def _talk_reads_for_checkpoint(result: dict) -> list[dict]:
    """Score the talk sentences with this cell's weights (skip if no checkpoint)."""
    from train_run import load_trained, predict_texts

    ckpt = result.get("checkpoint_dir")
    if not ckpt:
        return []
    try:
        model, tokenizer = load_trained(ckpt)
        return reads_from_predict(predict_texts(model, tokenizer, list(TALK_SENTENCES)))
    except Exception as exc:  # noqa: BLE001 — report still publishes without reads
        log.warning("talk-sentence classify skipped: %s", exc)
        return []


# @cpu_env.task: run this function as a Flyte task in the CPU environment.
#   cache="auto": if this task already ran with the same inputs and the same code,
#                 Flyte reuses the earlier output instead of running it again.
#   report=True:  the task gets an HTML report tab in the UI (here: the label mix).
@cpu_env.task(cache="auto", report=True)
async def prepare_data(split: str = cfg.DEFAULT_SPLIT) -> int:
    """Download Combined-S train/test xlsx for the paper's three seeds."""
    from load_data import download_paper_splits

    from report_stages import data_report

    paths = download_paper_splits(split=split, seeds=cfg.PAPER_SEEDS)
    log.info("FOMC splits ready: %s files under data/", len(paths))
    await flyte.report.replace.aio(data_report(split=split, seed=cfg.DEFAULT_SEED))
    await flyte.report.flush.aio()
    return len(paths)


# @gpu_env.task: run this function as a Flyte task in the GPU environment.
#   report=True: the task gets a report in the UI. Here it has a "Training" tab
#                with the live validation curve, redrawn after every epoch.
#   retries=2:   if the task fails (a preempted node, a driver hiccup, an OOM),
#                Flyte runs it again, up to 2 more times. The paper loop saves a
#                checkpoint every epoch (loop_checkpoint.pt), so a retry resumes
#                from the last epoch instead of starting over (force=True always
#                starts clean).
@gpu_env.task(report=True, retries=2)
async def train_one(
    model_key: str = "roberta-base",
    method: str = "lora",
    seed: int = cfg.DEFAULT_SEED,
    split: str = cfg.DEFAULT_SPLIT,
    max_epochs: int | None = None,
    force: bool = False,
) -> CellOut:
    """One paper-loop cell. Skips a finished metrics.json unless force=True.

    Training runs in a worker thread so this task can redraw the "Training"
    report tab after every epoch (read from the loop's progress.jsonl).
    """
    from train_run import run_one

    # Train in a background thread; meanwhile, every 15 s, redraw the Training
    # tab from the epochs the loop has logged to progress.jsonl.
    progress = cfg.run_dir(model_key, method, int(seed), split) / "progress.jsonl"
    training = asyncio.create_task(
        asyncio.to_thread(
            run_one,
            model_key=model_key,
            method=method,
            seed=int(seed),
            split=split,
            max_epochs=max_epochs,
            resume=not force,
            force=force,
        )
    )
    shown = 0
    while not training.done():
        await asyncio.wait({training}, timeout=15)
        shown = await _publish_training_curve(progress, model_key, method, int(seed), shown)
    result = training.result()
    await _publish_training_curve(progress, model_key, method, int(seed), -1, result=result)
    reads = _talk_reads_for_checkpoint(result)
    if reads:
        result["talk_reads"] = reads
        persist_talk_reads(result, reads)
    grid = refresh_report_snapshot(reads=reads or None, talk_source=source_label(result) if reads else None)
    highlight = (str(result.get("model")), str(result.get("method")))
    await _flush_talk_report(
        fomc_talk_report(result=result, reads=reads or None, grid=grid),
        reads=reads or None,
        grid=grid,
        highlight=highlight,
    )
    return _cell_out(result, reads)


# GPU task with retries, no report. It returns a flyte.io.Dir: Flyte uploads the
# merged model folder and passes a reference to it, which vllm_app (flyte_env.py)
# then serves.
@gpu_env.task(retries=2)
async def merge_adapter(
    method: str = "qlora",
    seed: int = cfg.DEFAULT_SEED,
    split: str = cfg.DEFAULT_SPLIT,
    force: bool = False,
) -> flyte.io.Dir:
    """Fold a trained Nemotron LoRA / QLoRA adapter into the BF16 base for vLLM.

    The returned directory is what `vllm_app` (flyte_env.py) serves. An existing
    merge is reused unless force=True.
    """
    from serve_vllm import merge_adapter as merge
    from serve_vllm import merged_dir

    out = merged_dir(method, int(seed), split)
    if force or not (out / "model.safetensors").exists():
        out = merge(method=method, seed=int(seed), split=split)
    return await flyte.io.Dir.from_local(out)


@dataclass
class ReleaseOut:
    """Named Summary -> Output for the release gate."""

    approved: bool
    seed: int
    model: str
    method: str
    test_f1: float
    test_f1_mean: float
    paper_f1: float
    merged_dir: str


def _release_html(rows: list[dict], model_key: str, method: str, mean: float, paper: float,
                  status: str, seed: int, split: str) -> str:
    from report_stages import load_predictions, release_report

    preds = load_predictions(cfg.run_dir(model_key, method, int(seed), split))
    return release_report(rows, model_key, method, mean, paper, status, preds=preds or None, shown_seed=int(seed))


# The release gate: a CPU task that shows the evidence in its report, then pauses
# for a person to decide (flyte.new_condition below) before merging an adapter.
@cpu_env.task(report=True)
async def release(
    model_key: str = "nemotron-nano-4b",
    method: str = "qlora",
    split: str = cfg.DEFAULT_SPLIT,
) -> ReleaseOut:
    """A person picks which seed's adapter to ship (or none), looking at the evidence.

    Locally the question is a prompt in the terminal; on a cluster it is an input
    box on the paused run in the Flyte UI. The chosen adapter is then merged for vLLM.
    """

    def _mine(runs: list[dict]) -> list[dict]:
        return [
            r for r in runs
            if r.get("model") == model_key and r.get("method") == method and r.get("split") == split
        ]

    # Checkpoints on this worker first; otherwise the results snapshot bundled with
    # the code (a fresh devbox / cluster pod has no checkpoints/ directory).
    rows = _mine(load_completed_runs())
    if not rows:
        from report_helpers import _previous_snapshot_runs

        rows = _mine(_previous_snapshot_runs())
    if not rows:
        raise ValueError(f"No finished {model_key} {method} runs on {split}. Train first.")
    rows = sorted(rows, key=lambda r: int(r["seed"]))
    mean = sum(float(r["weighted_f1"]) for r in rows) / len(rows)
    paper = float(cfg.paper_f1("roberta-large", split) or 0.0)
    by_seed = {int(r["seed"]): float(r["weighted_f1"]) for r in rows}
    best = max(by_seed, key=by_seed.get)
    await flyte.report.replace.aio(_release_html(rows, model_key, method, mean, paper, "Waiting for a decision...", best, split))
    await flyte.report.flush.aio()

    # flyte.new_condition pauses the run until someone answers. With --local it
    # asks in the terminal; on a devbox or cluster the paused run shows an input
    # box in the UI. If nobody answers within 24 hours, the task times out.
    choices = ", ".join(f"{s} (F1 {f:.3f})" for s, f in by_seed.items())
    gate = await flyte.new_condition.aio(
        "pick_seed_to_ship",
        prompt=(
            f"{cfg.display_name(model_key)} {method}: mean test F1 {mean:.3f} vs RoBERTa-large {paper:.3f}. "
            f"Which seed's adapter should ship? {choices}. Type a seed, or 'none'."
        ),
        data_type=str,
        timeout=timedelta(hours=24),
    )
    answer = str(await gate.wait.aio()).strip().lower()
    seed = int(answer) if answer.isdigit() and int(answer) in by_seed else -1
    merged = ""
    if seed < 0:
        status = "Not approved. Nothing merged or deployed." if answer in {"none", "no", "n", ""} else (
            f"'{answer}' is not one of the seeds; nothing merged or deployed.")
        shown = best
    else:
        shown = seed
        adapter = cfg.run_dir(model_key, method, seed, split) / "final" / "backbone" / "adapter_config.json"
        if not adapter.exists():
            status = (
                f"Approved seed {seed}, but its adapter is not on this worker. Run "
                f"`flyte run --local workflow.py merge_adapter --seed {seed}` where the checkpoint lives."
            )
        else:
            # flyte.group only affects the UI: the steps inside are shown together
            # under a "serve" heading. Calling a task from a task runs it as a child step.
            with flyte.group("serve"):
                out = await merge_adapter(method=method, seed=seed, split=split)
            merged = str(out.path)
            status = (f"Approved seed {seed}. Merged adapter ready for vLLM: {merged}. "
                      "Deploy with `flyte deploy workflow.py vllm_app`.")
    await flyte.report.replace.aio(_release_html(rows, model_key, method, mean, paper, status, shown, split))
    await flyte.report.flush.aio()
    return ReleaseOut(
        approved=seed >= 0, seed=seed, model=model_key, method=method,
        test_f1=round(by_seed.get(seed, 0.0), 4), test_f1_mean=round(mean, 4), paper_f1=paper, merged_dir=merged,
    )


# Not part of the talk: a placeholder for the market-returns half of the paper.
@cpu_env.task(report=True)
async def fetch_market_data() -> str:
    """
    Stub for the market half of Trillion Dollar Words.

    Next: pull QQQ / Treasury series and overlay the hawkish–dovish measure
    to replicate their Figure 2 (short QQQ when hawkish, long when dovish).
    Same Flyte CPU task pattern as prepare_data — extra financial files, not
    a rewrite of the classifier.
    """
    cfg.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "stub",
        "paper_figure": "Figure 2 — $100 portfolio, hawkish short QQQ / dovish long QQQ",
        "next": [
            "Download daily QQQ (and optional 2y/10y Treasury) for the paper window",
            "Score FOMC release days with a trained checkpoint",
            "Build the long/short series and compare to buy-and-hold",
        ],
    }
    path = cfg.OUTPUTS_DIR / "market_data_TODO.json"
    path.write_text(json.dumps(payload, indent=2))
    log.info("market-data stub wrote %s", path)
    await flyte.report.replace.aio(market_data_report(str(path)))
    await flyte.report.flush.aio()
    return str(path)


# A driver task: it does no heavy work itself, it calls other tasks. Each
# `await some_task(...)` below becomes a child step in the UI, running in that
# task's own environment (prepare_data on CPU, train_one on GPU).
@cpu_env.task(report=True)
async def pipeline(
    model_key: str = "roberta-base",
    method: str = "lora",
    seed: int = cfg.DEFAULT_SEED,
    split: str = cfg.DEFAULT_SPLIT,
    max_epochs: int | None = None,
    force: bool = False,
    include_market_stub: bool = False,
) -> CellOut:
    """Prepare Combined-S data, train one cell, optionally register the market stub."""
    log.info("FOMC pipeline: %s %s seed=%s", model_key, method, seed)
    await prepare_data(split=split)
    cell = await train_one(
        model_key=model_key,
        method=method,
        seed=seed,
        split=split,
        max_epochs=max_epochs,
        force=force,
    )
    if include_market_stub:
        await fetch_market_data()
    summary = await snapshot_grid()
    await write_talk_report(summary)
    return cell


# The 3-seed grid as one driver task: one train_one child step per
# (model, method, seed). Steps are grouped by model in the UI.
@cpu_env.task(report=True)
async def grid(
    split: str = cfg.DEFAULT_SPLIT,
    max_epochs: int | None = None,
    force: bool = False,
    include_qlora: bool | None = None,
    include_market_stub: bool = False,
    parallel: str = "serial",
    seeds: list[int] | None = None,
    include_nemotron: bool = False,
    include_baselines: bool = False,
) -> GridOut:
    """Paper 3-seed grid. CUDA runs all six cells (base+large × full/LoRA/QLoRA).

    parallel=serial trains one cell at a time (what a single GPU needs).
    parallel=jobs starts every train_one at once, so a cluster with several
    GPUs can give each its own GPU. include_nemotron adds Nemotron 3 Nano 4B
    QLoRA; include_baselines adds the no-fine-tuning row per model.
    """
    import shutil

    await prepare_data(split=split)
    if include_qlora is None:
        include_qlora = shutil.which("nvidia-smi") is not None
    jobs: list[tuple[str, str]] = []
    for model_key, method in cfg.ALL_JOBS:
        if method == "qlora" and not include_qlora:
            continue
        jobs.append((model_key, method))
    if include_nemotron:
        jobs.append(("nemotron-nano-4b", "qlora"))
    if include_baselines:  # the pretrained model with no fine-tuning, per model in the grid
        jobs = [(m, "zeroshot") for m in dict.fromkeys(m for m, _ in jobs)] + jobs
    mode = (parallel or "serial").strip().lower()
    if mode not in {"serial", "jobs"}:
        raise ValueError("parallel must be 'serial' or 'jobs'")
    seed_list = [int(x) for x in (seeds or cfg.PAPER_SEEDS)]
    cells = [
        (seed, model_key, method)
        for seed in seed_list
        for model_key, method in jobs
    ]
    total = len(cells)
    _say(f"START grid  {total} cells  jobs={jobs}  seeds={seed_list}  parallel={mode}")

    async def _run_cell(index: int, seed: int, model_key: str, method: str) -> CellOut:
        _say(f"START cell {index}/{total}: {model_key} {method} seed={seed}")
        cell = await train_one(
            model_key=model_key,
            method=method,
            seed=seed,
            split=split,
            max_epochs=max_epochs,
            force=force,
        )
        _announce_cell(cell, index, total, split, jobs)
        return cell

    numbered = list(enumerate(cells, 1))
    models = list(dict.fromkeys(model_key for _, model_key, _ in cells))

    async def _run_group(model_key: str) -> None:
        mine = [(i, c) for i, c in numbered if c[1] == model_key]
        # flyte.group: show this model's cells together under one heading in the UI.
        # asyncio.gather starts the child tasks at the same time; Flyte runs them in parallel.
        with flyte.group(cfg.display_name(model_key)):
            if mode == "jobs":
                await asyncio.gather(*[_run_cell(i, *c) for i, c in mine])
            else:
                for i, c in mine:
                    await _run_cell(i, *c)

    if mode == "jobs":
        await asyncio.gather(*[_run_group(m) for m in models])
    else:
        for m in models:
            await _run_group(m)
    _say(f"DONE grid  {total}/{total} cells")
    if include_market_stub:
        await fetch_market_data()
    summary = await snapshot_grid()
    return await write_talk_report(summary)


# Report-side tasks below run in the slim ui_env (no torch, no GPU). They read
# finished results; they do not train.
@ui_env.task
async def cell_summary(model: str, method: str, seed: int) -> CellOut:
    """One finished paper-loop cell — Input/Output on Summary, no training."""
    row = None
    for item in load_completed_runs():
        if (
            str(item.get("model")) == model
            and str(item.get("method")) == method
            and int(item.get("seed") or 0) == seed
        ):
            row = item
            break
    if row is None:
        return CellOut(
            model=model,
            method=method,
            seed=seed,
            weighted_f1=-1.0,
            vs_paper=0.0,
            device="",
            status="pending",
            labels="",
        )
    f1 = row.get("weighted_f1")
    paper = cfg.paper_f1(model)
    talks, src = load_talk_reads()
    labels = ""
    if src and model in src and method in src and str(seed) in src:
        labels = _label_csv(talks)
    return CellOut(
        model=model,
        method=method,
        seed=seed,
        weighted_f1=float(f1) if f1 is not None else -1.0,
        vs_paper=(float(f1) - paper) if f1 is not None and paper is not None else 0.0,
        device=str(row.get("device") or ""),
        status=str(row.get("status") or "complete"),
        labels=labels,
    )


@ui_env.task(report=True)
async def snapshot_grid() -> GridOut:
    """Load finished cells + sentence labels (metrics.json or bundled snapshot)."""
    from report_stages import grid_report

    grid = refresh_report_snapshot()
    await flyte.report.replace.aio(grid_report(grid["runs"]))
    await flyte.report.flush.aio()
    talks, _src = load_talk_reads()
    return _grid_out(grid, talks)


@ui_env.task(report=True)
async def write_talk_report(summary: GridOut) -> GridOut:
    """Report tab for this action. Input is the GridOut from snapshot_grid."""
    grid = summarize_grid()
    talks, _src = load_talk_reads()
    await _flush_talk_report(
        fomc_talk_report(grid=grid, reads=talks or None),
        reads=talks or None,
        grid=grid,
    )
    return summary


# The run shown in the talk: builds the comparison report ("Why fine-tune?")
# from finished results, with one child step per finished run.
@ui_env.task(report=True)
async def main() -> GridOut:
    """Driver: snapshot, one action per finished cell, then the talk Report on this run."""
    summary = await snapshot_grid()
    runs = load_completed_runs()
    if runs:
        await asyncio.gather(
            *[
                cell_summary(str(r["model"]), str(r["method"]), int(r["seed"]))
                for r in runs
            ]
        )
    out = await write_talk_report(summary)
    # Same HTML on the driver so the run-level Reports tab is the talk report
    # (not only the write_talk_report child).
    grid = summarize_grid()
    talks, _src = load_talk_reads()
    await _flush_talk_report(
        fomc_talk_report(grid=grid, reads=talks or None),
        reads=talks or None,
        grid=grid,
    )
    return out


@ui_env.task(report=True)
async def publish_talk_report() -> GridOut:
    """Same as main — kept so older CLI lines still work."""
    return await main()


# `python workflow.py`: connect using the Flyte config file and run `main`.
if __name__ == "__main__":
    flyte.init_from_config()
    r = flyte.run(main)
    print(r.name)
    print(r.url)
    r.wait()


