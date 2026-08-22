#!/usr/bin/env python3
"""Evaluate every controlled candidate with the v0.3 directional evaluator."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


EXPERIMENTS = (
    "motion_direct",
    "scale_separated_gated",
    "lstsq_gated",
    "appearance_gated",
    "motion_consistency",
    "motion_uncertainty",
)


def run(args: argparse.Namespace) -> dict:
    results = []
    for name in EXPERIMENTS:
        checkpoint = args.experiment_root / name / "best.pt"
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        split_results = {}
        for split in ("validation", "test"):
            output = args.experiment_root / name / f"directional_{split}.json"
            command = [
                sys.executable,
                "scripts/evaluate_directional_residual.py",
                "--checkpoint",
                str(checkpoint),
                "--index",
                str(args.index),
                "--split",
                split,
                "--samples",
                str(args.samples),
                "--output",
                str(output),
                "--device",
                args.device,
                "--batch-size",
                str(args.batch_size),
            ]
            completed = subprocess.run(command, check=False)
            if completed.returncode != 0:
                raise SystemExit(f"evaluation failed: {name} {split}")
            split_results[split] = json.loads(output.read_text(encoding="utf-8"))["summary"]
        results.append({"name": name, "validation": split_results["validation"], "test": split_results["test"]})
    report = {"samples": args.samples, "experiments": results}
    output = args.experiment_root / "directional_matrix_summary.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--experiment-root", type=Path, default=Path("artifacts/v030_experiments"))
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    report = run(args)
    print(json.dumps({"experiments": len(report["experiments"]), "samples": report["samples"]}, indent=2))


if __name__ == "__main__":
    main()
