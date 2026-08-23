#!/usr/bin/env python3
"""Evaluate the unchanged V1-A field on normal-phone raster contracts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flyrot.data.tartanair import default_split
from flyrot.models.flyrot_v1 import FlyRotV1
from flyrot.real_texture import RealTextureRotationDataset, manifest_sha256, path_free_record
from scripts.run_t4t5_correctness_audit import _evaluate_field


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:1" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit-per-split", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20260823)
    args = parser.parse_args()
    device = torch.device(args.device)
    records = json.loads(args.index.read_text(encoding="utf-8"))
    splits = default_split(records)
    source_manifest = {name: [path_free_record(item) for item in values] for name, values in splits.items()}
    model = FlyRotV1(scales=(4, 8, 16)).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model"] if "model" in state else state)
    model.eval()
    results: dict[str, object] = {}
    for width, height in ((192, 108), (256, 144), (320, 180)):
        raster_name = f"{width}x{height}"
        results[raster_name] = {}
        for window in (3, 5, 9, 17):
            results[raster_name][str(window)] = {}
            for split_name in ("train", "validation", "test"):
                dataset = RealTextureRotationDataset(
                    splits[split_name],
                    window_length=window,
                    image_size=(width, height),
                    limit=args.limit_per_split,
                    seed=args.seed,
                )
                metrics = _evaluate_field(model, dataset, device, collect_panel=False)
                metrics.pop("panel", None)
                results[raster_name][str(window)][split_name] = {
                    "dataset_manifest_sha256": manifest_sha256(dataset.manifest()),
                    "metrics": metrics,
                }
    report = {
        "schema": "flyrot.phone-motion-field-audit.v1",
        "commit": __import__("subprocess").check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "checkpoint_sha256": __import__("hashlib").sha256(args.checkpoint.read_bytes()).hexdigest(),
        "source_split_manifest_sha256": manifest_sha256(source_manifest),
        "device": str(device),
        "rasters": [[192, 108], [256, 144], [320, 180]],
        "windows": [3, 5, 9, 17],
        "selection_split": "validation only; test reported after selection and not used for selection",
        "model_frontend": "unchanged V1-A historical P0",
        "results": results,
        "status": "failed_until_field_gate_passes",
        "claim_boundary": "phone-raster motion-field audit; no rotation, translation, or metric-scale acceptance claim",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    summary = {}
    for raster, windows in results.items():
        summary[raster] = {
            window: windows[window]["validation"]["metrics"]["polarity"]["combined"]
            for window in windows
        }
    print(json.dumps({"output": str(args.output), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
