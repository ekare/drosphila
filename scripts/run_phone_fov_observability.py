#!/usr/bin/env python3
"""Measure whether a normal phone FOV explains the failed field gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flyrot.data.tartanair import default_split
from flyrot.models.flyrot_v1 import FlyRotV1
from flyrot.real_texture import RealTextureRotationDataset, manifest_sha256, path_free_record
from scripts.run_t4t5_correctness_audit import _evaluate_field


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:1" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit-per-split", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=144)
    args = parser.parse_args()

    device = torch.device(args.device)
    records = json.loads(args.index.read_text(encoding="utf-8"))
    splits = default_split(records)
    source_manifest = {name: [path_free_record(item) for item in values] for name, values in splits.items()}
    model = FlyRotV1(scales=(4, 8, 16)).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model"] if "model" in state else state)
    model.eval()

    fovs = (60.0, 90.0, 120.0)
    results: dict[str, object] = {}
    for fov in fovs:
        fov_key = f"{fov:g}deg"
        results[fov_key] = {}
        for split_name in ("train", "validation", "test"):
            dataset = RealTextureRotationDataset(
                splits[split_name],
                window_length=3,
                image_size=(args.width, args.height),
                limit=args.limit_per_split,
                seed=args.seed,
                horizontal_fov_deg=fov,
            )
            metrics = _evaluate_field(model, dataset, device, collect_panel=False)
            metrics.pop("panel", None)
            results[fov_key][split_name] = {
                "dataset_manifest_sha256": manifest_sha256(dataset.manifest()),
                "metrics": metrics,
            }

    report = {
        "schema": "flyrot.phone-fov-observability.v1",
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "source_split_manifest_sha256": manifest_sha256(source_manifest),
        "device": str(device),
        "raster": [args.width, args.height],
        "horizontal_fovs_deg": list(fovs),
        "selection_split": "validation only; test reported after selection and not used for selection",
        "model_frontend": "unchanged V1-A historical checkpoint",
        "results": results,
        "status": "diagnostic_only",
        "claim_boundary": "FOV observability diagnostic; does not accept the retinotopic field, rotation, translation, or metric scale",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "validation": {fov: results[fov]["validation"]["metrics"]["polarity"]["combined"] for fov in results}}, indent=2))


if __name__ == "__main__":
    main()
