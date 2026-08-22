#!/usr/bin/env python3
"""Audit the analytic SO(3) solver against exact dense rotational flow."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import torch

from flyrot.data.tartanair import default_split
from flyrot.geometry.camera import tartanair_v2_lcam_front_intrinsics
from flyrot.geometry.so3 import exp_so3, log_so3
from flyrot.real_texture import RealTextureRotationDataset, manifest_sha256, solve_weighted_rotation


def commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def run(dataset: RealTextureRotationDataset, device: torch.device) -> dict[str, float | int | None]:
    intrinsics = tartanair_v2_lcam_front_intrinsics(640, 640).resized(8, 8)
    correct: list[float] = []
    wrong: list[float] = []
    conditions: list[float] = []
    for index in range(len(dataset)):
        item = dataset[index]
        flow = item["dense_flow"].unsqueeze(0).to(device)
        valid = item["valid_mask"].unsqueeze(0).to(device)
        b, steps, _, height, width = flow.shape
        flow_4d = flow.reshape(b * steps, 2, height, width)
        valid_4d = valid.reshape(b * steps, 1, height, width)
        field = torch.nn.functional.adaptive_avg_pool2d(flow_4d * valid_4d, (8, 8)) / torch.nn.functional.adaptive_avg_pool2d(valid_4d, (8, 8)).clamp_min(1e-8)
        field = field.reshape(b, steps, 2, 8, 8)
        field[:, :, 0] *= 8.0 / float(width)
        field[:, :, 1] *= 8.0 / float(height)
        weights = torch.nn.functional.adaptive_avg_pool2d(valid_4d, (8, 8)).reshape(b, steps, 8, 8)
        solved = solve_weighted_rotation(field, weights, intrinsics)
        target = item["target_step_rotation_vectors"].unsqueeze(0).to(device)
        predicted = solved["rotation_vector"]
        correct.extend(torch.rad2deg(torch.linalg.vector_norm(predicted - target, dim=-1)).cpu().reshape(-1).tolist())
        wrong_target = -target
        wrong.extend(torch.rad2deg(torch.linalg.vector_norm(predicted - wrong_target, dim=-1)).cpu().reshape(-1).tolist())
        conditions.extend(solved["condition"].cpu().reshape(-1).tolist())
    return {"samples": len(dataset), "mean_correct_error_deg": float(np.mean(correct)), "median_correct_error_deg": float(np.median(correct)), "mean_wrong_sign_error_deg": float(np.mean(wrong)), "wrong_minus_correct_deg": float(np.mean(wrong) - np.mean(correct)), "median_condition": float(np.median(conditions)), "valid": True}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:1" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--limit-per-split", type=int, default=48)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    records = json.loads(args.index.read_text(encoding="utf-8"))
    splits = default_split(records)
    report = {"experiment": {"name": "exact-dense-flow-analytic-so3-solver", "seed": args.seed, "device": str(device), "field_raster": [8, 8], "checkpoint_sha256": None}, "commit": commit(), "results": {}}
    for window in (3, 5, 7):
        dataset = RealTextureRotationDataset(splits["validation"], window_length=window, image_size=(128, 128), limit=args.limit_per_split, seed=args.seed)
        report["results"][str(window)] = {"dataset_manifest_sha256": manifest_sha256(dataset.manifest()), "validation": run(dataset, device)}
    (args.output_dir / "geometric_so3_solver.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (args.output_dir / "geometric_so3_solver.md").write_text("# Geometric SO(3) solver audit\n\nExact dense rotational flow was downsampled to the 8x8 field and solved with bounded IRLS. Correct-sign and inverse-sign errors are reported for T=3,5,7. This validates the geometry solver contract only; it does not validate the learned local field.\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
