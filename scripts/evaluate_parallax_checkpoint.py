#!/usr/bin/env python3
"""Evaluate a checkpoint by depth-measured translational-flow contamination."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from flyrot.data.tartanair import TartanAirWindowDataset
from flyrot.models.flyrot_v0 import FlyRotV0
from flyrot.models.tiny_conv_baseline import TinyConvBaseline


def make_model(name: str, device: str) -> torch.nn.Module:
    if name == "flyrot":
        return FlyRotV0().to(device)
    if name == "flyrot_gated":
        return FlyRotV0(confidence_gated=True, readout_scale=8.0).to(device)
    if name == "flyrot_motion_direct":
        return FlyRotV0(scales=(4, 8, 16), readout_scale=1.0, magnitude_confidence=True).to(device)
    if name == "flyrot_motion_consistency":
        return FlyRotV0(scales=(4, 8, 16), readout_scale=1.0, motion_consistency_gate=True).to(device)
    if name == "tiny_conv_32":
        return TinyConvBaseline(channels=32).to(device)
    raise ValueError(f"unsupported model for parallax evaluation: {name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--model",
        choices=("flyrot", "flyrot_gated", "flyrot_motion_direct", "flyrot_motion_consistency", "tiny_conv_32"),
        default="flyrot",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--window-length", type=int, default=2)
    parser.add_argument("--frame-gap", type=int)
    parser.add_argument(
        "--target-direction",
        choices=("camera_relative", "image_motion", "optical_relative", "optical_image_motion"),
        default="camera_relative",
    )
    args = parser.parse_args()
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    records = json.loads(args.index.read_text(encoding="utf-8"))
    record_map = {(r["environment"], r["trajectory"]): r for r in records}
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    model = make_model(args.model, args.device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    groups: dict[str, list[float]] = {"low": [], "medium": [], "high": []}
    zero_groups: dict[str, list[float]] = {"low": [], "medium": [], "high": []}
    pair_results = []
    datasets = {}
    with torch.no_grad():
        for item in audit["pairs"]:
            key = (item["environment"], item["trajectory"])
            if key not in datasets:
                dataset = TartanAirWindowDataset(
                    [record_map[key]],
                    window_length=args.window_length,
                    frame_gap=args.frame_gap or audit["gap"],
                    image_size=args.image_size,
                    target_direction=args.target_direction,
                )
                datasets[key] = dataset
            dataset = datasets[key]
            start = int(item["start"])
            sample = dataset[start]
            frames = sample["frames"].unsqueeze(0).to(args.device)
            target = sample["target_rotation_vector"].unsqueeze(0).to(args.device)
            prediction = model(frames)["rotation_vector"][:, -1]
            model_mse = float((prediction - target).square().mean().cpu())
            zero_mse = float(target.square().mean().cpu())
            ratio = float(item["translation_to_full_ratio_median"])
            group = "low" if ratio < 0.25 else "medium" if ratio < 0.50 else "high"
            groups[group].append(model_mse)
            zero_groups[group].append(zero_mse)
            pair_results.append(
                {
                    "environment": item["environment"],
                    "trajectory": item["trajectory"],
                    "start": start,
                    "parallax_ratio": ratio,
                    "group": group,
                    "model_mse": model_mse,
                    "zero_mse": zero_mse,
                }
            )

    def mean(values: list[float]) -> float | None:
        return sum(values) / len(values) if values else None

    summary = {
        group: {
            "pairs": len(groups[group]),
            "model_mse": mean(groups[group]),
            "zero_mse": mean(zero_groups[group]),
            "model_below_zero": bool(groups[group] and mean(groups[group]) < mean(zero_groups[group])),
        }
        for group in groups
    }
    result = {
        "checkpoint": str(args.checkpoint),
        "model": args.model,
        "audit": str(args.audit),
        "summary": summary,
        "pairs": pair_results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
