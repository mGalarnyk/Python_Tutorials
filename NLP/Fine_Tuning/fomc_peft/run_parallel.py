"""1 GPU/job: unfinished RoBERTa cells, one process per card.

Does not do 4 GPUs/model (DDP). paper_loop is single-device. Finished
metrics.json is skipped unless --force.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

import torch

import config as cfg
from train_run import log


def unfinished(
    models: list[str],
    methods: list[str],
    force: bool,
) -> list[tuple[str, str, int]]:
    cells: list[tuple[str, str, int]] = []
    for seed in cfg.PAPER_SEEDS:
        for model in models:
            for method in methods:
                path = cfg.run_dir(model, method, int(seed)) / "metrics.json"
                if force or not path.exists():
                    cells.append((model, method, int(seed)))
    return cells


def main() -> int:
    parser = argparse.ArgumentParser(description="1 GPU/job parallel RoBERTa cells")
    parser.add_argument("--models", default="roberta-base,roberta-large")
    parser.add_argument("--methods", default="full,lora,qlora")
    parser.add_argument("--workers", type=int, default=0, help="0 = all visible GPUs")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    n_gpu = torch.cuda.device_count()
    if n_gpu < 1:
        log("no GPU")
        return 1
    workers = args.workers or n_gpu
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    cells = unfinished(models, methods, args.force)
    log(f"1 GPU/job  gpus={n_gpu}  workers={workers}  remaining={len(cells)}")
    if not cells:
        log("nothing left — finished metrics.json already on disk")
        return 0

    python = sys.executable
    running: dict[int, subprocess.Popen] = {}
    next_gpu = 0
    failed = 0
    for i, (model, method, seed) in enumerate(cells, start=1):
        while len(running) >= workers:
            for gpu, proc in list(running.items()):
                code = proc.poll()
                if code is None:
                    continue
                running.pop(gpu)
                if code != 0:
                    failed += 1
                    log(f"worker gpu={gpu} exited {code}")
            if len(running) >= workers:
                time.sleep(2)

        while next_gpu in running:
            next_gpu = (next_gpu + 1) % n_gpu
        gpu = next_gpu
        next_gpu = (next_gpu + 1) % n_gpu
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        cmd = [
            python,
            "train_run.py",
            "--model",
            model,
            "--method",
            method,
            "--seed",
            str(seed),
        ]
        if args.force:
            cmd.append("--force")
        log(f"START {i}/{len(cells)}  gpu={gpu}  {model} {method} seed={seed}")
        running[gpu] = subprocess.Popen(cmd, env=env, cwd=str(cfg.ROOT))

    for gpu, proc in running.items():
        code = proc.wait()
        if code != 0:
            failed += 1
            log(f"worker gpu={gpu} exited {code}")
    log(f"DONE 1 GPU/job  failed={failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
