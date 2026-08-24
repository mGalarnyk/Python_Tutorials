"""Compare full / LoRA / QLoRA on Union's text-to-SQL pipeline. Results go in the notebook."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd

from device_utils import describe, has_nvidia
from run_profile import flyte_local_argv, recommend

ROOT = Path(__file__).resolve().parent
OUTPUTS_DIR = ROOT / "outputs"
RUNS_PATH = OUTPUTS_DIR / "grid_runs.csv"
SUMMARY_PATH = OUTPUTS_DIR / "grid_summary.csv"


def recommend_jobs(include_qlora: bool | None = None) -> list[str]:
    """Mac: full + LoRA. CUDA: also QLoRA (bitsandbytes 4-bit)."""
    if include_qlora is None:
        include_qlora = has_nvidia()
    jobs = ["full", "lora"]
    if include_qlora:
        jobs.append("qlora")
    return jobs


def metrics_path(method: str) -> Path:
    return OUTPUTS_DIR / f"run_{method}.json"


def collect_completed() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if not OUTPUTS_DIR.exists():
        return pd.DataFrame()
    for path in sorted(OUTPUTS_DIR.glob("run_*.json")):
        data = json.loads(path.read_text())
        rows.append(
            {
                "who": "ours (Union pipeline, Mac/CUDA adapted)",
                "model": data.get("model_name"),
                "method": data.get("method", path.stem.replace("run_", "")),
                "base_accuracy": data.get("base_accuracy"),
                "finetuned_accuracy": data.get("finetuned_accuracy"),
                "improvement": data.get("improvement"),
                "num_examples": data.get("num_examples"),
                "epochs": data.get("epochs"),
                "max_train_samples": data.get("max_train_samples"),
                "device": data.get("device"),
                "trainable_pct": data.get("trainable_pct"),
            }
        )
    return pd.DataFrame(rows)


def format_results(runs: pd.DataFrame) -> pd.DataFrame:
    if runs.empty:
        return runs
    view = runs.copy()
    for col in ("base_accuracy", "finetuned_accuracy", "improvement"):
        if col in view:
            view[col] = pd.to_numeric(view[col], errors="coerce")
    cols = [
        "who",
        "model",
        "method",
        "base_accuracy",
        "finetuned_accuracy",
        "improvement",
        "num_examples",
        "epochs",
        "device",
        "trainable_pct",
    ]
    return view[[c for c in cols if c in view.columns]].reset_index(drop=True)


def load_runs() -> pd.DataFrame:
    runs = collect_completed()
    if not runs.empty:
        return runs
    if RUNS_PATH.exists():
        return pd.read_csv(RUNS_PATH)
    raise FileNotFoundError(
        "No finished text-to-SQL runs yet. Run the grid cell, or one flyte pipeline first."
    )


def run_grid(
    jobs: list[str] | None = None,
    smoke: bool = True,
    resume: bool = True,
) -> pd.DataFrame:
    """Run Union's pipeline once per method. Skip a method if its metrics JSON exists."""
    profile = recommend(smoke=smoke)
    jobs = jobs or recommend_jobs(include_qlora=profile["qlora_ok"])
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[sql] grid on {describe()}  jobs={jobs}  smoke={smoke}", flush=True)

    for i, method in enumerate(jobs, start=1):
        out = metrics_path(method)
        if resume and out.exists():
            print(f"[sql] grid cell {i}/{len(jobs)}: {method} already complete — {out}", flush=True)
            continue
        print(f"[sql] grid cell {i}/{len(jobs)}: {method}", flush=True)
        cmd = flyte_local_argv(profile, method=method)
        print("[sql]", " ".join(cmd), flush=True)
        subprocess.run(cmd, cwd=ROOT, check=True)

    runs = collect_completed()
    if not runs.empty:
        runs.to_csv(RUNS_PATH, index=False)
        runs.to_csv(SUMMARY_PATH, index=False)
    return format_results(runs)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Union text-to-SQL full / LoRA / QLoRA grid")
    parser.add_argument("--no-smoke", action="store_true", help="Union-scale sample counts (RTX PRO 6000 Blackwell)")
    parser.add_argument("--force", action="store_true", help="re-run even if run_<method>.json exists")
    parser.add_argument("--include-qlora", action="store_true")
    args = parser.parse_args()
    table = run_grid(
        jobs=recommend_jobs(include_qlora=True if args.include_qlora else None),
        smoke=not args.no_smoke,
        resume=not args.force,
    )
    print(table.to_string(index=False) if not table.empty else "no runs yet")
    print(f"NVIDIA / bitsandbytes QLoRA: {has_nvidia()}")
