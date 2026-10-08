"""Run the FOMC comparison grid: full vs LoRA vs QLoRA, ours vs Shah et al.

Default path is Flyte 2 (`flyte run --local workflow.py pipeline` per cell),
same orchestration as part 1. Pass use_flyte=False / --no-flyte for the
in-process paper loop only.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Sequence
from typing import Any

import pandas as pd

import config as cfg
from device_utils import detect_device, describe, has_nvidia
from paper_baselines import MAC_TEST_F1, MAC_TRAIN, eval_released_checkpoint, paper_table
from train_run import log, run_one

RUNS_PATH = cfg.OUTPUTS_DIR / "grid_runs.csv"
SUMMARY_PATH = cfg.OUTPUTS_DIR / "grid_summary.csv"
SUMMARY_JSON = cfg.OUTPUTS_DIR / "grid_summary.json"


def recommend_jobs(
    include_qlora: bool | None = None,
    include_large_full: bool | None = None,
) -> list[tuple[str, str]]:
    """Mac: base+large full and LoRA. CUDA: the full 2×3 (also both QLoRA cells)."""
    device = detect_device()
    if include_qlora is None:
        include_qlora = device == "cuda"
    if include_large_full is None:
        include_large_full = True

    jobs: list[tuple[str, str]] = []
    for model_key, method in cfg.ALL_JOBS:
        if method == "qlora" and not include_qlora:
            continue
        if model_key == "roberta-large" and method == "full" and not include_large_full:
            continue
        jobs.append((model_key, method))
    return jobs


def flyte_pipeline_argv(
    model_key: str,
    method: str,
    seed: int,
    split: str = cfg.DEFAULT_SPLIT,
    max_epochs: int | None = None,
    force: bool = False,
    include_market_stub: bool = False,
    tui: bool = False,
) -> list[str]:
    """Flyte 2 local CLI for one paper-loop cell. Flags use underscores.

    `tui=False` here because run_grid() shells this out from a notebook.
    Interactive: `flyte run --local --tui workflow.py pipeline ...`
    """
    if method == "qlora" and not has_nvidia():
        raise RuntimeError("QLoRA needs NVIDIA CUDA. On this laptop use lora or full.")
    cmd = [
        "flyte",
        "run",
        "--local",
    ]
    if tui:
        cmd.append("--tui")
    cmd.extend([
        "workflow.py",
        "pipeline",
        "--model_key",
        model_key,
        "--method",
        method,
        "--seed",
        str(int(seed)),
        "--split",
        split,
    ])
    # Plain `bool` task inputs are on/off flags in the flyte CLI (no value).
    if include_market_stub:
        cmd.append("--include_market_stub")
    if max_epochs is not None:
        cmd.extend(["--max_epochs", str(max_epochs)])
    if force:
        cmd.append("--force")
    return cmd


def flyte_grid_argv(
    split: str = cfg.DEFAULT_SPLIT,
    max_epochs: int | None = None,
    force: bool = False,
    include_qlora: bool | None = None,
) -> list[str]:
    cmd = [
        "flyte",
        "run",
        "--local",
        "--tui",
        "workflow.py",
        "grid",
        "--split",
        split,
        "--include_market_stub",
    ]
    if max_epochs is not None:
        cmd.extend(["--max_epochs", str(max_epochs)])
    if force:
        cmd.append("--force")
    if include_qlora is True:
        cmd.extend(["--include_qlora", "true"])
    elif include_qlora is False:
        cmd.extend(["--include_qlora", "false"])
    return cmd


def collect_completed(split: str = cfg.DEFAULT_SPLIT) -> pd.DataFrame:
    """Finished paper-loop runs from `checkpoints/*/metrics.json` (skips old Trainer dirs)."""
    rows: list[dict[str, Any]] = []
    if not cfg.CHECKPOINTS_DIR.exists():
        return pd.DataFrame()
    for metrics_path in sorted(cfg.CHECKPOINTS_DIR.glob("*/metrics.json")):
        if not cfg.is_tracked_run_dir(metrics_path.parent.name):
            continue
        data = json.loads(metrics_path.read_text())
        if data.get("status") != "complete":
            continue
        if data.get("split", split) != split:
            continue
        rows.append(_row_from_result(data, who="ours"))
    return pd.DataFrame(rows)


def _run_identity(row: dict[str, Any]) -> tuple[str, str, int, str]:
    return (
        str(row.get("model") or ""),
        str(row.get("method") or ""),
        int(row.get("seed") or 0),
        machine_label(row.get("device"), str(row.get("who") or "ours")),
    )


def _mac_published_runs(split: str = cfg.DEFAULT_SPLIT) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (model, method), f1s in MAC_TEST_F1.items():
        epochs, minutes = MAC_TRAIN[(model, method)]
        for seed, f1 in zip(cfg.PAPER_SEEDS, f1s):
            rows.append(
                {
                    "who": "ours",
                    "model": model,
                    "method": method,
                    "seed": int(seed),
                    "split": split,
                    "weighted_f1": float(f1),
                    "epochs_trained": float(epochs),
                    "wall_seconds": float(minutes) * 60.0,
                    "device": "mps",
                    "status": "complete",
                }
            )
    return pd.DataFrame(rows)


def _snapshot_runs() -> pd.DataFrame:
    """Bundled CUDA (or mixed) runs. Display fallback only, not for skip-finished."""
    try:
        from report_snapshot import SNAPSHOT
    except ImportError:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for rec in SNAPSHOT.get("runs", []):
        if rec.get("status") != "complete":
            continue
        rows.append(
            {
                "who": "ours",
                "model": rec.get("model"),
                "method": rec.get("method"),
                "seed": rec.get("seed"),
                "split": rec.get("split", cfg.DEFAULT_SPLIT),
                "weighted_f1": rec.get("weighted_f1"),
                "epochs_trained": rec.get("epochs_trained"),
                "trainable_pct": rec.get("trainable_pct"),
                "wall_seconds": rec.get("wall_seconds"),
                "device": rec.get("device"),
                "status": "complete",
            }
        )
    return pd.DataFrame(rows)


def merge_report_runs(runs: pd.DataFrame, split: str = cfg.DEFAULT_SPLIT) -> pd.DataFrame:
    """Union live checkpoints with the CUDA snapshot and Mac 3-seed F1s.

    Training skip logic still uses collect_completed() (disk only). This merge
    is for the Results plot/table so Mac and CUDA both show when only one
    machine's checkpoints are on disk.
    """
    frames = [df for df in (runs, _snapshot_runs(), _mac_published_runs(split)) if df is not None and not df.empty]
    if not frames:
        return pd.DataFrame()
    seen: set[tuple[str, str, int, str]] = set()
    kept: list[dict[str, Any]] = []
    for frame in frames:
        for rec in frame.to_dict(orient="records"):
            key = _run_identity(rec)
            if key in seen:
                continue
            seen.add(key)
            kept.append(rec)
    return pd.DataFrame(kept)


def typical_minutes(seconds: pd.Series) -> float | None:
    """Mean wall time, dropping seeds that sat with the lid closed (>> fastest seed)."""
    mins = pd.to_numeric(seconds, errors="coerce").dropna() / 60.0
    if mins.empty:
        return None
    fastest = float(mins.min())
    cap = max(fastest * 2.5, fastest + 15.0)
    kept = mins[mins <= cap]
    if kept.empty:
        return float(mins.median())
    return float(kept.mean())


def machine_label(device: str | None = None, who: str = "") -> str:
    """Notebook Machine column. Name the SKU, not a vague GPU."""
    who_l = (who or "").lower()
    if "shah" in who_l and "released" not in who_l:
        return "RTX A6000"
    d = (device or "").lower()
    if "rtx pro 5000" in d:
        return "RTX PRO 5000 Blackwell laptop, 24 GB, CUDA"
    if d == "cuda" or "rtx" in d or "nvidia" in d or "h200" in d:
        return "RTX PRO 6000 Blackwell, 96 GB, CUDA"
    if d == "mps" or "mps" in d:
        return "MacBook Pro M4 Max, 128 GB, MPS"
    if d == "cpu":
        return "CPU"
    return device or "—"


def format_results(summary: pd.DataFrame) -> pd.DataFrame:
    """Notebook table: 3-seed mean ± std vs Shah et al. Table 5."""

    def f1_cell(row: pd.Series) -> str:
        mean = row.get("weighted_f1")
        std = row.get("weighted_f1_std")
        if mean is None or (isinstance(mean, float) and pd.isna(mean)):
            return "—"
        n = row.get("n_seeds") or 0
        if n > 1 and std is not None and not pd.isna(std):
            return f"{float(mean):.3f} ± {float(std):.3f}"
        return f"{float(mean):.3f}"

    def delta_cell(row: pd.Series) -> str:
        who = str(row.get("who") or "")
        if who.startswith("Shah"):
            return "—"
        d = row.get("delta_vs_paper")
        if d is None or (isinstance(d, float) and pd.isna(d)):
            return "—"
        return f"{float(d):+.3f}"

    def pct_cell(row: pd.Series) -> str:
        p = row.get("trainable_pct")
        if p is None or (isinstance(p, float) and pd.isna(p)):
            return "—"
        return f"{float(p):.1f}%"

    def epochs_cell(row: pd.Series) -> str:
        e = row.get("epochs_trained_mean")
        if e is None or (isinstance(e, float) and pd.isna(e)):
            return "—"
        return f"{float(e):.0f}"

    def time_cell(row: pd.Series) -> str:
        m = row.get("train_minutes")
        if m is None or (isinstance(m, float) and pd.isna(m)):
            return "—"
        m = float(m)
        if m >= 120:
            return f"{m / 60:.1f} h"
        return f"{m:.0f} min" if m >= 10 else f"{m:.1f} min"

    def per_epoch_cell(row: pd.Series) -> str:
        s = row.get("sec_per_epoch")
        if s is None or (isinstance(s, float) and pd.isna(s)):
            return "—"
        s = float(s)
        if s >= 3600:
            return f"{s / 3600:.1f} h"
        if s >= 90:
            return f"{s / 60:.1f} min"
        if s >= 20:
            return f"{s:.0f} s"
        return f"{s:.1f} s"

    out = pd.DataFrame(
        {
            "who": summary["who"],
            "model": summary["model"],
            "method": summary["method"],
            "weighted F1": summary.apply(f1_cell, axis=1),
            "vs paper": summary.apply(delta_cell, axis=1),
            "epochs": summary.apply(epochs_cell, axis=1),
            "time": summary.apply(time_cell, axis=1),
            "time/epoch": summary.apply(per_epoch_cell, axis=1),
            "trainable": summary.apply(pct_cell, axis=1),
            "machine": summary["machine"]
            if "machine" in summary.columns
            else summary.apply(
                lambda row: machine_label(row.get("device"), str(row.get("who") or "")),
                axis=1,
            ),
            "status": summary["status"],
        }
    )
    return out.reset_index(drop=True)


def style_results_by_machine(df: pd.DataFrame):
    """Color notebook rows by the Machine column (same palette as the Flyte report)."""
    from report_helpers import machine_row_color

    if df is None or df.empty or "machine" not in df.columns:
        return df

    def _row(row: pd.Series) -> list[str]:
        bg = machine_row_color(str(row.get("machine") or ""))
        return [f"background-color: {bg}" for _ in row]

    return df.style.apply(_row, axis=1)


def format_per_seed(runs: pd.DataFrame) -> pd.DataFrame:
    """One row per (model, method, seed) for the notebook."""
    if runs.empty:
        return runs
    cols = [
        "model",
        "method",
        "seed",
        "weighted_f1",
        "accuracy",
        "epochs_trained",
        "wall_seconds",
        "device",
        "status",
    ]
    view = runs[[c for c in cols if c in runs.columns]].copy()
    if "device" in view.columns:
        view["machine"] = view["device"].map(lambda d: machine_label(d, "ours"))
    for col in ("weighted_f1", "accuracy"):
        if col in view:
            view[col] = pd.to_numeric(view[col], errors="coerce").round(4)
    if "epochs_trained" in view:
        view["epochs"] = pd.to_numeric(view["epochs_trained"], errors="coerce").round(0).astype("Int64")
        view = view.drop(columns=["epochs_trained"])
    if "wall_seconds" in view:
        view["minutes"] = (pd.to_numeric(view["wall_seconds"], errors="coerce") / 60).round(1)
        view = view.drop(columns=["wall_seconds"])
    return view.reset_index(drop=True)


def _write_outputs(runs: pd.DataFrame, split: str) -> pd.DataFrame:
    cfg.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    runs.to_csv(RUNS_PATH, index=False)
    summary = summarize_runs(runs, split=split)
    summary.to_csv(SUMMARY_PATH, index=False)
    from report_helpers import refresh_report_snapshot

    refresh_report_snapshot()
    return summary


def _row_from_result(result: dict[str, Any], who: str = "ours") -> dict[str, Any]:
    test = result.get("test") or {}
    return {
        "who": who,
        "model": result.get("model"),
        "method": result.get("method"),
        "seed": result.get("seed"),
        "split": result.get("split"),
        "weighted_f1": test.get("weighted_f1"),
        "accuracy": test.get("accuracy"),
        "macro_f1": test.get("macro_f1"),
        "f1_dovish": test.get("f1_dovish"),
        "f1_hawkish": test.get("f1_hawkish"),
        "f1_neutral": test.get("f1_neutral"),
        "paper_weighted_f1": result.get("paper_weighted_f1"),
        "delta_vs_paper": result.get("delta_vs_paper"),
        "trainable_params": result.get("trainable_params"),
        "trainable_pct": result.get("trainable_pct"),
        "peak_mem_gb": result.get("peak_mem_gb"),
        "wall_seconds": result.get("wall_seconds"),
        "epochs_trained": result.get("epochs_trained"),
        "checkpoint_dir": result.get("checkpoint_dir") or result.get("final_dir"),
        "device": result.get("device"),
        "status": result.get("status", "complete"),
    }


def summarize_runs(runs: pd.DataFrame, split: str = cfg.DEFAULT_SPLIT) -> pd.DataFrame:
    """Mean ± std over seeds, same aggregation as paper Table 5."""
    rows: list[dict[str, Any]] = []
    for rec in paper_table(split).to_dict(orient="records"):
        rows.append(
            {
                "who": rec["who"],
                "model": rec["model"],
                "method": rec["method"],
                "split": rec["split"],
                "n_seeds": 3,
                "seeds": ",".join(str(s) for s in cfg.PAPER_SEEDS),
                "weighted_f1": rec["weighted_f1"],
                "weighted_f1_std": rec["weighted_f1_std"],
                "paper_weighted_f1": rec["weighted_f1"],
                "delta_vs_paper": 0.0,
                "trainable_pct": 100.0,
                "machine": "RTX A6000",
                "status": "published",
            }
        )

    ours = runs[(runs["who"] == "ours") & (runs["status"] == "complete")].copy()
    if ours.empty:
        return pd.DataFrame(rows)
    ours["machine"] = ours["device"].map(lambda d: machine_label(d, "ours"))

    for (model, method, machine), group in ours.groupby(["model", "method", "machine"], sort=False):
        group = group.sort_values("seed")
        f1 = pd.to_numeric(group["weighted_f1"], errors="coerce").dropna()
        n = int(len(f1))
        paper = cfg.paper_f1(str(model), split)
        mean_f1 = float(f1.mean()) if n else None
        std_f1 = float(f1.std(ddof=1)) if n > 1 else 0.0
        seed_list = [int(s) for s in group["seed"].tolist()]

        def _mean(col: str) -> float | None:
            if col not in group.columns:
                return None
            vals = pd.to_numeric(group[col], errors="coerce").dropna()
            return float(vals.mean()) if len(vals) else None

        def _max(col: str) -> float | None:
            if col not in group.columns:
                return None
            vals = pd.to_numeric(group[col], errors="coerce").dropna()
            return float(vals.max()) if len(vals) else None

        def _first(col: str):
            if col not in group.columns:
                return None
            return group[col].iloc[0]

        train_minutes = (
            typical_minutes(group["wall_seconds"]) if "wall_seconds" in group.columns else None
        )
        epochs_mean = _mean("epochs_trained")
        sec_per_epoch = None
        if train_minutes and epochs_mean and float(epochs_mean) > 0:
            sec_per_epoch = float(train_minutes) * 60.0 / float(epochs_mean)

        rows.append(
            {
                "who": "ours",
                "model": model,
                "method": method,
                "split": split,
                "n_seeds": n,
                "seeds": ",".join(str(s) for s in seed_list),
                "f1_by_seed": ",".join(f"{v:.4f}" for v in f1.tolist()),
                "weighted_f1": mean_f1,
                "weighted_f1_std": std_f1,
                "accuracy": _mean("accuracy"),
                "macro_f1": _mean("macro_f1"),
                "paper_weighted_f1": paper,
                "delta_vs_paper": None if paper is None or mean_f1 is None else mean_f1 - paper,
                "trainable_params": _first("trainable_params"),
                "trainable_pct": _first("trainable_pct"),
                "peak_mem_gb": _max("peak_mem_gb"),
                "wall_seconds_mean": _mean("wall_seconds"),
                "wall_seconds_total": (
                    float(pd.to_numeric(group["wall_seconds"], errors="coerce").sum())
                    if "wall_seconds" in group.columns
                    else None
                ),
                "train_minutes": train_minutes,
                "epochs_trained_mean": epochs_mean,
                "sec_per_epoch": sec_per_epoch,
                "device": str(_first("device") or "") or None,
                "machine": machine,
                "status": "complete"
                if n >= len(cfg.PAPER_SEEDS)
                else f"partial ({n}/{len(cfg.PAPER_SEEDS)})",
            }
        )
    return pd.DataFrame(rows)


def run_grid(
    jobs: list[tuple[str, str]] | None = None,
    seeds: Sequence[int] | None = None,
    split: str = cfg.DEFAULT_SPLIT,
    max_epochs: int | None = None,
    resume: bool = True,
    force: bool = False,
    eval_released: bool = False,
    use_flyte: bool = True,
) -> pd.DataFrame:
    jobs = jobs or recommend_jobs()
    seeds = list(seeds or cfg.PAPER_SEEDS)
    cfg.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    cells = [(model, method, seed) for seed in seeds for model, method in jobs]
    log(
        f"grid on {describe()}  jobs={jobs}  seeds={seeds}  "
        f"split={split}  cells={len(cells)}  flyte={use_flyte}"
    )

    rows: list[dict[str, Any]] = []
    raw: list[dict[str, Any]] = []
    n = len(cells)
    launched = 0
    for i, (model_key, method, seed) in enumerate(cells, start=1):
        metrics_path = cfg.run_dir(model_key, method, int(seed), split) / "metrics.json"
        if resume and not force and metrics_path.exists():
            log(f"grid cell {i}/{n}: {model_key} {method} seed={seed} already complete")
            continue
        log(f"grid cell {i}/{n}: {model_key} {method} seed={seed}")
        try:
            if use_flyte:
                cmd = flyte_pipeline_argv(
                    model_key=model_key,
                    method=method,
                    seed=int(seed),
                    split=split,
                    max_epochs=max_epochs,
                    force=force,
                    include_market_stub=False,
                )
                log(" ".join(cmd))
                subprocess.run(cmd, cwd=cfg.ROOT, check=True)
                launched += 1
            else:
                result = run_one(
                    model_key=model_key,
                    method=method,
                    seed=int(seed),
                    split=split,
                    max_epochs=max_epochs,
                    resume=resume,
                    force=force,
                )
                raw.append(result)
                rows.append(_row_from_result(result, who="ours"))
            completed = collect_completed(split=split)
            if not completed.empty:
                _write_outputs(completed, split)
        except Exception as exc:
            log(f"grid cell {i}/{n} failed: {type(exc).__name__}: {exc}")
            rows.append(
                {
                    "who": "ours",
                    "model": model_key,
                    "method": method,
                    "seed": seed,
                    "split": split,
                    "status": f"failed: {exc}",
                }
            )

    if use_flyte and launched:
        stub = ["flyte", "run", "--local", "workflow.py", "fetch_market_data"]
        log(" ".join(stub))
        try:
            subprocess.run(stub, cwd=cfg.ROOT, check=True)
        except Exception as exc:
            log(f"market-data stub skipped: {exc}")

    if eval_released:
        for seed in seeds:
            try:
                released = eval_released_checkpoint(split=split, seed=int(seed))
                raw.append(released)
                rows.append(
                    _row_from_result(released, who="Shah et al. released checkpoint")
                )
            except Exception as exc:
                log(f"released checkpoint seed={seed} skipped: {exc}")
                rows.append(
                    {
                        "who": "Shah et al. released checkpoint",
                        "model": "roberta-large",
                        "method": "released",
                        "seed": seed,
                        "split": split,
                        "status": f"skipped: {exc}",
                    }
                )

    runs = collect_completed(split=split)
    if runs.empty:
        runs = pd.DataFrame(rows)
    summary = _write_outputs(runs, split)
    if not raw:
        raw = runs.to_dict(orient="records")
    SUMMARY_JSON.write_text(json.dumps(raw, indent=2, default=str))
    log(f"wrote per-seed {RUNS_PATH}")
    log(f"wrote 3-seed mean {SUMMARY_PATH}")
    return summary


def load_summary(split: str = cfg.DEFAULT_SPLIT) -> pd.DataFrame:
    """Mean table from finished paper-loop checkpoints (falls back to CSV)."""
    runs = collect_completed(split=split)
    if not runs.empty:
        return summarize_runs(runs, split=split)
    if SUMMARY_PATH.exists():
        return pd.read_csv(SUMMARY_PATH)
    raise FileNotFoundError(
        "No finished paper-loop runs yet. Run run_grid() or wait for metrics.json."
    )


def load_runs(split: str = cfg.DEFAULT_SPLIT) -> pd.DataFrame:
    """Per-seed table from finished paper-loop checkpoints (falls back to CSV)."""
    runs = collect_completed(split=split)
    if not runs.empty:
        return runs
    if RUNS_PATH.exists():
        return pd.read_csv(RUNS_PATH)
    raise FileNotFoundError(
        "No finished paper-loop runs yet. Run run_grid() or wait for metrics.json."
    )


def _parse_seeds(args) -> list[int]:
    if args.seed is not None:
        return [int(args.seed)]
    if args.seeds:
        return [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    return list(cfg.PAPER_SEEDS)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="FOMC full / LoRA / QLoRA comparison grid")
    parser.add_argument("--models", default=None, help="comma list: roberta-base,roberta-large")
    parser.add_argument("--methods", default=None, help="comma list: full,lora,qlora")
    parser.add_argument(
        "--seeds",
        default=None,
        help="comma list of seeds (default: paper 5768,78516,944601)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="run a single seed instead of the paper's three",
    )
    parser.add_argument("--split", default=cfg.DEFAULT_SPLIT)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--eval-released", action="store_true")
    parser.add_argument("--include-qlora", action="store_true")
    parser.add_argument("--include-large-full", action="store_true")
    parser.add_argument(
        "--no-flyte",
        action="store_true",
        help="in-process paper loop (skip flyte run --local)",
    )
    args = parser.parse_args()

    if args.models or args.methods:
        models = (args.models or "roberta-base,roberta-large").split(",")
        methods = (args.methods or "full,lora").split(",")
        jobs = [(m.strip(), h.strip()) for m in models for h in methods]
    else:
        jobs = recommend_jobs(
            include_qlora=True if args.include_qlora else None,
            include_large_full=True if args.include_large_full else None,
        )

    table = run_grid(
        jobs=jobs,
        seeds=_parse_seeds(args),
        split=args.split,
        max_epochs=args.max_epochs,
        resume=not args.no_resume,
        force=args.force,
        eval_released=args.eval_released,
        use_flyte=not args.no_flyte,
    )
    print(table.to_string(index=False))
    print(f"NVIDIA / QLoRA ok: {has_nvidia()}")
    print(f"per-seed: {RUNS_PATH}")
    print(f"3-seed mean: {SUMMARY_PATH}")
