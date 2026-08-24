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

import json
import logging
import os
from dataclasses import dataclass

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import flyte
import flyte.report  # required — `import flyte` does not load this submodule

import config as cfg
from flyte_env import cpu_env, gpu_env, ui_env
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


@gpu_env.task(report=True)
async def train_one(
    model_key: str = "roberta-base",
    method: str = "lora",
    seed: int = cfg.DEFAULT_SEED,
    split: str = cfg.DEFAULT_SPLIT,
    max_epochs: int | None = None,
    force: bool = False,
) -> CellOut:
    """One paper-loop cell. Skips a finished metrics.json unless force=True."""
    from train_run import run_one

    result = run_one(
        model_key=model_key,
        method=method,
        seed=int(seed),
        split=split,
        max_epochs=max_epochs,
        resume=not force,
        force=force,
    )
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
) -> GridOut:
    """Paper 3-seed grid. CUDA runs all six cells (base+large × full/LoRA/QLoRA).

    parallel=serial walks cells (safe on this Mac). parallel=jobs launches
    train_one concurrently so Flyte can place one GPU per unfinished cell.
    """
    import asyncio
    import shutil

    await prepare_data(split=split)
    if include_qlora is None:
        include_qlora = shutil.which("nvidia-smi") is not None
    jobs: list[tuple[str, str]] = []
    for model_key, method in cfg.ALL_JOBS:
        if method == "qlora" and not include_qlora:
            continue
        jobs.append((model_key, method))
    mode = (parallel or "serial").strip().lower()
    if mode not in {"serial", "jobs"}:
        raise ValueError("parallel must be 'serial' or 'jobs'")
    log.info("FOMC grid jobs=%s seeds=%s parallel=%s", jobs, list(cfg.PAPER_SEEDS), mode)
    coros = [
        train_one(
            model_key=model_key,
            method=method,
            seed=int(seed),
            split=split,
            max_epochs=max_epochs,
            force=force,
        )
        for seed in cfg.PAPER_SEEDS
        for model_key, method in jobs
    ]
    if mode == "jobs":
        await asyncio.gather(*coros)
    else:
        for coro in coros:
            await coro
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
    import asyncio

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


