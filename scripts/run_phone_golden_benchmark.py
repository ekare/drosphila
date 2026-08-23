#!/usr/bin/env python3
"""Run bounded normal-phone raster polarity and ray-space golden gates."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flyrot.geometry.phone_camera import camera_ray_grid
from flyrot.geometry.rotational_flow import CameraIntrinsics
from flyrot.models.direction_cells import DirectionCellBank
from flyrot.models.phone_frontend import PhonePolarityFrontend
from flyrot.t4t5 import StimulusSpec, make_stimulus
from flyrot.validation_contract import direction_index, photoreceptor_variants, pure_edge_stimulus, response_metrics

from scripts.run_t4t5_correctness_audit import _bank_pooled


def _commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _evaluate(frames: torch.Tensor, valid: torch.Tensor, variant: str, bank: DirectionCellBank, device: torch.device) -> dict[str, object]:
    frontend = PhonePolarityFrontend(variant).to(device).eval()
    with torch.inference_mode():
        result = frontend(frames.to(device))
        pooled = _bank_pooled(result["on"], result["off"], valid.to(device), bank)
    return {"pooled": pooled, "frontend": result}


def _portable_result(pooled: dict[str, torch.Tensor], expected: int, device: torch.device) -> dict[str, object]:
    metrics = response_metrics(pooled["response"].unsqueeze(0), torch.tensor([expected], device=device))
    return {
        "metrics": metrics,
        "on_peak": float(pooled["on_response"].max()),
        "off_peak": float(pooled["off_response"].max()),
        "response_peak": float(pooled["response"].max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:1" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rasters = ((320, 180), (256, 144), (192, 108))
    scales = (1, 2, 4, 8)
    rows = []
    for width, height in rasters:
        intrinsics = CameraIntrinsics(fx=0.88 * width, fy=0.90 * height, cx=0.51 * width, cy=0.48 * height, width=width, height=height)
        ray_grid = camera_ray_grid(intrinsics)
        bank = DirectionCellBank(scales=scales).to(device).eval()
        for variant in PhonePolarityFrontend.VARIANTS:
            on_frames, on_valid = pure_edge_stimulus("on", angle_deg=45, displacement=4, width=width, height=height)
            off_frames, off_valid = pure_edge_stimulus("off", angle_deg=45, displacement=4, width=width, height=height)
            on_result = _evaluate(on_frames.unsqueeze(0), on_valid, variant, bank, device)
            off_result = _evaluate(off_frames.unsqueeze(0), off_valid, variant, bank, device)
            expected = direction_index(45)
            matched_on = float(on_result["pooled"]["on_response"].max())
            leakage_on = float(on_result["pooled"]["off_response"].max()) / max(matched_on, 1e-8)
            matched_off = float(off_result["pooled"]["off_response"].max())
            leakage_off = float(off_result["pooled"]["on_response"].max()) / max(matched_off, 1e-8)
            flicker_frames, flicker_valid = make_stimulus(StimulusSpec("flicker", height=height, width=width, steps=5))
            stationary_frames, stationary_valid = make_stimulus(StimulusSpec("stationary_uniform", height=height, width=width, steps=5))
            flicker = _evaluate(flicker_frames.unsqueeze(0), flicker_valid, variant, bank, device)["pooled"]
            stationary = _evaluate(stationary_frames.unsqueeze(0), stationary_valid, variant, bank, device)["pooled"]
            rows.append(
                {
                    "raster": [width, height],
                    "intrinsics": intrinsics.as_dict(),
                    "ray_valid_fraction": float(ray_grid.valid.float().mean()),
                    "variant": variant,
                    "on_edge": _portable_result(on_result["pooled"], expected, device),
                    "off_edge": _portable_result(off_result["pooled"], expected, device),
                    "on_edge_off_leakage": leakage_on,
                    "off_edge_on_leakage": leakage_off,
                    "flicker_ratio": float(flicker["response"].max()) / max((matched_on + matched_off) * 0.5, 1e-8),
                    "stationary_ratio": float(stationary["response"].max()) / max((matched_on + matched_off) * 0.5, 1e-8),
                }
            )

    summary = {}
    for variant in PhonePolarityFrontend.VARIANTS:
        selected = [row for row in rows if row["variant"] == variant]
        on_leak = float(np.median([row["on_edge_off_leakage"] for row in selected]))
        off_leak = float(np.median([row["off_edge_on_leakage"] for row in selected]))
        flicker = float(np.median([row["flicker_ratio"] for row in selected]))
        stationary = float(np.median([row["stationary_ratio"] for row in selected]))
        direction_errors = [
            row["on_edge"]["metrics"]["population_vector_median_error_deg"]
            for row in selected
            if row["on_edge"]["metrics"]["population_vector_median_error_deg"] is not None
        ] + [
            row["off_edge"]["metrics"]["population_vector_median_error_deg"]
            for row in selected
            if row["off_edge"]["metrics"]["population_vector_median_error_deg"] is not None
        ]
        summary[variant] = {
            "on_to_off_leakage": on_leak,
            "off_to_on_leakage": off_leak,
            "flicker_ratio": flicker,
            "stationary_ratio": stationary,
            "ideal_direction_median_error_deg": float(np.median(direction_errors)) if direction_errors else None,
            "p0_gate": bool(on_leak <= 0.15 and off_leak <= 0.15 and flicker <= 0.10 and stationary <= 0.05 and direction_errors and np.median(direction_errors) <= 5.0),
        }
    report = {
        "schema": "flyrot.phone-golden-benchmark.v1",
        "commit": _commit(),
        "device": str(device),
        "production_camera": "single_normal_perspective_phone_camera",
        "rasters": [list(item) for item in rasters],
        "cell_scales": list(scales),
        "rows": rows,
        "summary": summary,
        "selection_split": "deterministic validation-style golden stimuli; no test split",
        "claim_boundary": "T4/T5-inspired engineering gate, not biological equivalence",
    }
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
