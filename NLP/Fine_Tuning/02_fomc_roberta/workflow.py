"""
FOMC hawkish–dovish Flyte 2 pipeline.

Same paper loop as run_grid.py / paper_loop.py. Flyte is the orchestration:
the same tasks run on this MacBook Pro (`flyte run --local`, MPS) and on a
remote NVIDIA RTX PRO 6000 Blackwell (`flyte run`, CUDA). CPU tasks fetch
data; GPU tasks train. Later CPU tasks can pull market series and replicate
Shah et al.'s QQQ returns chart (Figure 2).

Usage:
    flyte run --local --tui workflow.py pipeline --model_key roberta-base --method lora --seed 5768
    flyte run --local --tui workflow.py grid
    flyte start tui                                         # browse persisted local runs
    # Serve a fine-tuned Nemotron with vLLM (OpenAI-compatible API):
    flyte run workflow.py merge_adapter --method qlora --seed 5768
    flyte deploy workflow.py vllm_app
    FLYTE_GPUS=1 flyte run workflow.py grid                 # one GPU, cells in series
    FLYTE_GPUS=1 flyte run workflow.py grid --parallel jobs # 1 GPU/job on a 4-GPU node
    # 4 GPUs/model (DDP) is a Results placeholder; the loop is still single-device.
    # Same Flyte 2 UI locally and on the GPU: Sub actions, Input, Output, Reports.
    #   Mac (CPU):  flyte start devbox
    #   Remote NVIDIA RTX PRO 6000 Blackwell:  flyte start devbox --gpu
    #   then (no --local):  flyte -c .flyte/devbox.yaml run workflow.py pipeline ...
    #   open http://localhost:30080/v2
    # Reports: report=True + import flyte.report (required).
    #   flyte -c .flyte/devbox.yaml run --name fomc-main workflow.py main
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
    prepare_data_report,
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

def _line_svg(xs: list[int], ys: list[float], title: str, color: str, best: int | None) -> str:
    """Small inline SVG line chart, no plotting deps (works in the slim report image)."""
    w, h, pad = 420, 220, 40
    lo, hi = min(ys), max(ys)
    span = (hi - lo) or 1.0
    x_max = max(max(xs), 2)

    def px(x: int) -> float:
        return pad + (x - 1) / (x_max - 1) * (w - 2 * pad)

    def py(y: float) -> float:
        return h - pad - (y - lo) / span * (h - 2 * pad)

    pts = " ".join(f"{px(x):.1f},{py(y):.1f}" for x, y in zip(xs, ys))
    dots = "".join(f'<circle cx="{px(x):.1f}" cy="{py(y):.1f}" r="3" fill="{color}"/>' for x, y in zip(xs, ys))
    mark = ""
    if best is not None:
        bx, by = px(xs[best]), py(ys[best])
        mark = (
            f'<circle cx="{bx:.1f}" cy="{by:.1f}" r="7" fill="none" stroke="#111" stroke-width="2"/>'
            f'<text x="{bx + 9:.1f}" y="{by - 9:.1f}" font-size="12">best {ys[best]:.3f} (epoch {xs[best]})</text>'
        )
    return (
        f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" style="font-family:sans-serif;margin:4px 12px 4px 0">'
        f'<text x="{pad}" y="20" font-size="14" font-weight="600">{title}</text>'
        f'<line x1="{pad}" y1="{h - pad}" x2="{w - pad}" y2="{h - pad}" stroke="#999"/>'
        f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{h - pad}" stroke="#999"/>'
        f'<text x="{pad - 6}" y="{py(hi) + 4:.1f}" font-size="11" text-anchor="end">{hi:.2f}</text>'
        f'<text x="{pad - 6}" y="{py(lo) + 4:.1f}" font-size="11" text-anchor="end">{lo:.2f}</text>'
        f'<text x="{w / 2:.0f}" y="{h - 8}" font-size="11" text-anchor="middle">epoch</text>'
        f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2"/>{dots}{mark}</svg>'
    )


def _training_curve_html(evals: list[dict], model_key: str, method: str, seed: int) -> str:
    xs = [int(e["epoch"]) for e in evals]
    f1 = [float(e["val_f1"]) for e in evals]
    ce = [float(e["val_ce"]) for e in evals]
    best = max(range(len(f1)), key=f1.__getitem__)
    last = evals[-1]
    return (
        f"<h3>{cfg.display_name(model_key)} · {method} · seed {seed}</h3>"
        f"<p>Epoch {last['epoch']}: val F1 {float(last['val_f1']):.3f}, val loss {float(last['val_ce']):.3f}, "
        f"early-stop counter {last.get('early_stopping_count', 0)}/7. "
        "Test scores come from the best-F1 epoch, not the last one.</p>"
        + _line_svg(xs, f1, "Validation weighted F1", "#2563eb", best)
        + _line_svg(xs, ce, "Validation loss (cross-entropy)", "#dc2626", None)
    )


async def _publish_training_curve(progress: Path, model_key: str, method: str, seed: int, shown: int) -> int:
    """Redraw the Training curve tab when progress.jsonl has new epochs. shown=-1 forces a redraw."""
    if not progress.exists():
        return shown
    evals = [r for r in map(json.loads, progress.read_text().splitlines()) if r.get("event") == "eval"]
    if not evals or (shown >= 0 and len(evals) == shown):
        return shown
    flyte.report.get_tab("Training curve").replace(_training_curve_html(evals, model_key, method, seed))
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


@cpu_env.task(cache="auto", report=True)
async def prepare_data(split: str = cfg.DEFAULT_SPLIT) -> int:
    """Download Combined-S train/test xlsx for the paper's three seeds."""
    from load_data import download_paper_splits

    paths = download_paper_splits(split=split, seeds=cfg.PAPER_SEEDS)
    log.info("FOMC splits ready: %s files under data/", len(paths))
    await flyte.report.replace.aio(
        prepare_data_report(len(paths), split, list(cfg.PAPER_SEEDS))
    )
    await flyte.report.flush.aio()
    return len(paths)


# retries: a preempted node, a driver hiccup, or an OOM reruns the task. The paper
# loop checkpoints every epoch (loop_checkpoint.pt), so a retry resumes there
# instead of starting over (unless force=True, which always starts clean).
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

    Training runs in a worker thread so this task can redraw the "Training curve"
    report tab after every epoch (read from the loop's progress.jsonl).
    """
    from train_run import run_one

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
    await _publish_training_curve(progress, model_key, method, int(seed), -1)
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
    model: str
    method: str
    test_f1_mean: float
    seeds: int
    paper_f1: float
    merged_dir: str


def _release_report(rows: list[dict], model_key: str, method: str, mean: float, paper: float, status: str) -> str:
    body = "".join(
        f"<tr><td>{r['seed']}</td><td>{float(r['weighted_f1']):.3f}</td><td>{int(float(r.get('epochs_trained') or 0))}</td></tr>"
        for r in rows
    )
    return (
        f"<h3>Release gate: {cfg.display_name(model_key)} · {method}</h3>"
        f"<p>Test weighted F1 <b>{mean:.3f}</b> over {len(rows)} seed(s) vs Shah et al. RoBERTa-large "
        f"<b>{paper:.3f}</b> (full fine-tune, Table 5).</p>"
        f"<table><tr><th>seed</th><th>test F1</th><th>epochs</th></tr>{body}</table>"
        f"<p><b>{status}</b></p>"
    )


@cpu_env.task(report=True)
async def release(
    model_key: str = "nemotron-nano-4b",
    method: str = "qlora",
    seed: int = cfg.DEFAULT_SEED,
    split: str = cfg.DEFAULT_SPLIT,
) -> ReleaseOut:
    """Human approval before serving: show the held-out F1, wait for yes/no, then merge.

    Locally the question is a [y/n] prompt in the terminal; on a cluster it is an
    Approve button on the run in the Flyte UI. `seed` picks which adapter to ship.
    """
    rows = [
        r for r in load_completed_runs()
        if r.get("model") == model_key and r.get("method") == method and r.get("split") == split
    ]
    if not rows:
        raise ValueError(f"No finished {model_key} {method} runs on {split}. Train first.")
    mean = sum(float(r["weighted_f1"]) for r in rows) / len(rows)
    paper = float(cfg.paper_f1("roberta-large", split) or 0.0)
    await flyte.report.replace.aio(_release_report(rows, model_key, method, mean, paper, "Waiting for approval..."))
    await flyte.report.flush.aio()

    gate = await flyte.new_condition.aio(
        "approve_release",
        prompt=(
            f"{cfg.display_name(model_key)} {method}: test F1 {mean:.3f} over {len(rows)} seed(s) "
            f"vs RoBERTa-large {paper:.3f}. Merge seed {seed} and serve with vLLM?"
        ),
        data_type=bool,
        timeout=timedelta(hours=24),
    )
    approved = bool(await gate.wait.aio())
    merged = ""
    if approved:
        with flyte.group("serve"):
            out = await merge_adapter(method=method, seed=int(seed), split=split)
        merged = str(out.path)
        status = f"Approved. Merged adapter ready for vLLM: {merged}. Deploy with `flyte deploy workflow.py vllm_app`."
    else:
        status = "Not approved. Nothing merged or deployed."
    await flyte.report.replace.aio(_release_report(rows, model_key, method, mean, paper, status))
    await flyte.report.flush.aio()
    return ReleaseOut(
        approved=approved, model=model_key, method=method, test_f1_mean=round(mean, 4),
        seeds=len(rows), paper_f1=paper, merged_dir=merged,
    )


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
    log.info(
        "FOMC pipeline: %s %s seed=%s  (local MPS or remote RTX PRO 6000 Blackwell)",
        model_key,
        method,
        seed,
    )
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
) -> GridOut:
    """Paper 3-seed grid. CUDA runs all six cells (base+large × full/LoRA/QLoRA).

    parallel=serial walks cells (safe on this Mac). parallel=jobs launches
    train_one concurrently so Flyte can place one GPU per unfinished cell.
    Cells are grouped per model in the UI / TUI. include_nemotron adds
    Nemotron-3-Nano-4B QLoRA (the method that fits a 24 GB card).
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


@ui_env.task
async def snapshot_grid() -> GridOut:
    """Load finished cells + sentence labels (metrics.json or bundled snapshot)."""
    grid = refresh_report_snapshot()
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


if __name__ == "__main__":
    flyte.init_from_config()
    r = flyte.run(main)
    print(r.name)
    print(r.url)
    r.wait()


