#!/usr/bin/env python3
"""Run a fixed, trajectory-disjoint v0.3 model comparison matrix.

The matrix deliberately keeps data split, sampling, seed, budget, loss, and
input size fixed. It writes only local generated run artifacts; the public
selection report is assembled separately from the resulting metrics.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


EXPERIMENTS = (
    ("motion_direct", "flyrot_motion_direct"),
    ("scale_separated_gated", "flyrot_scale_separated_gated"),
    ("lstsq_gated", "flyrot_lstsq_gated"),
    ("appearance_gated", "flyrot_appearance_gated"),
    ("motion_consistency", "flyrot_motion_consistency"),
    ("motion_uncertainty", "flyrot_motion_uncertainty"),
)


def run(args: argparse.Namespace) -> dict:
    root = args.output_root
    root.mkdir(parents=True, exist_ok=True)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.call(["git", "diff", "--quiet"]) != 0
    common = [
        sys.executable,
        "scripts/train.py",
        "--index",
        str(args.index),
        "--steps",
        str(args.steps),
        "--eval-interval",
        str(args.eval_interval),
        "--max-train-samples",
        str(args.max_train_samples),
        "--max-val-samples",
        str(args.max_val_samples),
        "--image-size",
        str(args.image_size),
        "--window-length",
        str(args.window_length),
        "--frame-gap",
        str(args.frame_gap),
        "--target-direction",
        args.target_direction,
        "--batch-size",
        str(args.batch_size),
        "--device",
        args.device,
        "--preload-images",
        "--preload-workers",
        str(args.preload_workers),
        "--seed",
        str(args.seed),
    ]
    summary = {
        "git_commit": commit,
        "working_tree_dirty_at_start": dirty,
        "common_config": {
            "split_policy": "default_split",
            "index_role": "local fast-cache index supplied by operator",
            "steps": args.steps,
            "eval_interval": args.eval_interval,
            "max_train_samples": args.max_train_samples,
            "max_val_samples": args.max_val_samples,
            "image_size": args.image_size,
            "window_length": args.window_length,
            "frame_gap": args.frame_gap,
            "target_direction": args.target_direction,
            "batch_size": args.batch_size,
            "device": args.device,
            "preload_images": True,
            "preload_workers": args.preload_workers,
            "seed": args.seed,
        },
        "experiments": [],
    }
    for name, model in EXPERIMENTS:
        run_dir = root / name
        metrics_path = run_dir / "metrics.json"
        entry = {"name": name, "model": model, "run_dir": str(run_dir), "command_model": model}
        command = [*common, "--run-dir", str(run_dir), "--model", model]
        entry["command"] = command
        started = time.monotonic()
        run_dir.mkdir(parents=True, exist_ok=True)
        with (run_dir / "console.log").open("w", encoding="utf-8") as log:
            completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
        entry["returncode"] = completed.returncode
        entry["runtime_seconds"] = time.monotonic() - started
        if metrics_path.exists():
            entry["metrics"] = json.loads(metrics_path.read_text(encoding="utf-8"))
        summary["experiments"].append(entry)
        (root / "matrix_summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
        if completed.returncode != 0:
            raise SystemExit(f"experiment failed: {name} (see {run_dir / 'console.log'})")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/v030_experiments_kiousb"))
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--max-train-samples", type=int, default=1024)
    parser.add_argument("--max-val-samples", type=int, default=256)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--window-length", type=int, default=3)
    parser.add_argument("--frame-gap", type=int, default=4)
    parser.add_argument("--target-direction", default="optical_image_motion")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--preload-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2, default=str))


if __name__ == "__main__":
    main()
