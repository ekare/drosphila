#!/usr/bin/env python3
"""Evaluate a FlyRot checkpoint on a trajectory-disjoint dataset split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from flyrot.data.tartanair import TartanAirWindowDataset, default_split
from flyrot.models.flyrot_v0 import FlyRotV0
from flyrot.models.tiny_conv_baseline import TinyConvBaseline


def evenly_spaced_indices(length: int, limit: int) -> list[int]:
    if limit >= length:
        return list(range(length))
    if limit <= 1:
        return [0]
    return sorted({round(index * (length - 1) / (limit - 1)) for index in range(limit)})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--index", type=Path, default=Path("artifacts/tartanair2_index.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--samples-per-record", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--image-height", type=int)
    parser.add_argument("--window-length", type=int, default=3)
    parser.add_argument("--frame-gap", type=int, default=4)
    parser.add_argument(
        "--target-direction",
        choices=("camera_relative", "image_motion", "optical_relative", "optical_image_motion"),
        default="camera_relative",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--preload-images", action="store_true")
    parser.add_argument("--preload-workers", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    image_size = (args.image_size, args.image_height) if args.image_height else args.image_size

    with args.index.open(encoding="utf-8") as handle:
        records = json.load(handle)
    records = default_split(records)[args.split]
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    model_name = checkpoint.get("config", {}).get("model", "flyrot")
    if model_name == "flyrot_gated":
        model = FlyRotV0(confidence_gated=True, readout_scale=8.0).to(args.device)
    elif model_name == "flyrot_magnitude_gated":
        model = FlyRotV0(confidence_gated=True, readout_scale=8.0, magnitude_aware=True).to(args.device)
    elif model_name == "flyrot_appearance_gated":
        model = FlyRotV0(confidence_gated=True, readout_scale=8.0, appearance_normalized=True).to(args.device)
    elif model_name == "flyrot_scaled_gated":
        model = FlyRotV0(scales=(2, 4, 8), confidence_gated=True, readout_scale=8.0).to(args.device)
    elif model_name == "flyrot_motion_scaled_gated":
        model = FlyRotV0(scales=(4, 8, 16), confidence_gated=True, readout_scale=8.0).to(args.device)
    elif model_name == "flyrot_scale_separated_gated":
        model = FlyRotV0(
            scales=(4, 8, 16),
            confidence_gated=True,
            readout_scale=8.0,
            scale_separated=True,
            magnitude_aware=True,
        ).to(args.device)
    elif model_name == "flyrot_lstsq_gated":
        model = FlyRotV0(scales=(4, 8, 16), confidence_gated=True, readout_scale=8.0, least_squares_basis=True).to(args.device)
    elif model_name == "flyrot_motion_direct":
        model = FlyRotV0(scales=(4, 8, 16), readout_scale=1.0, magnitude_confidence=True).to(args.device)
    elif model_name == "flyrot_motion_consistency":
        model = FlyRotV0(scales=(4, 8, 16), readout_scale=1.0, motion_consistency_gate=True).to(args.device)
    elif model_name == "flyrot_motion_uncertainty":
        model = FlyRotV0(scales=(4, 8, 16), readout_scale=1.0, input_dependent_uncertainty=True).to(args.device)
    elif model_name == "flyrot_stretched_gated":
        model = FlyRotV0(confidence_gated=True, readout_scale=8.0, focal_y_over_x=4.0 / 3.0).to(args.device)
    elif model_name == "tiny_conv_32":
        model = TinyConvBaseline(channels=32).to(args.device)
    else:
        model = (FlyRotV0() if model_name == "flyrot" else TinyConvBaseline()).to(args.device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    per_record = []
    all_prediction_errors = []
    all_zero_errors = []
    for record in records:
        dataset = TartanAirWindowDataset(
            [record],
            window_length=args.window_length,
            frame_gap=args.frame_gap,
            image_size=image_size,
            target_direction=args.target_direction,
            preload_images=args.preload_images,
            preload_workers=args.preload_workers,
        )
        limit = min(args.samples_per_record, len(dataset))
        selected = torch.utils.data.Subset(dataset, evenly_spaced_indices(len(dataset), limit))
        loader = DataLoader(selected, batch_size=args.batch_size, shuffle=False, num_workers=0)
        prediction_errors = []
        zero_errors = []
        with torch.no_grad():
            seen = 0
            for batch in loader:
                if seen >= limit:
                    break
                take = min(len(batch["frames"]), limit - seen)
                frames = batch["frames"][:take].to(args.device)
                target = batch["target_rotation_vector"][:take].to(args.device)
                prediction = model(frames)["rotation_vector"][:, -1]
                prediction_errors.append((prediction - target).square().mean(dim=1).cpu())
                zero_errors.append(target.square().mean(dim=1).cpu())
                seen += take
        prediction_values = torch.cat(prediction_errors).tolist() if prediction_errors else []
        zero_values = torch.cat(zero_errors).tolist() if zero_errors else []
        all_prediction_errors.extend(prediction_values)
        all_zero_errors.extend(zero_values)
        per_record.append(
            {
                "environment": record["environment"],
                "trajectory": record["trajectory"],
                "samples": len(prediction_values),
                "model_mse": sum(prediction_values) / len(prediction_values) if prediction_values else None,
                "zero_mse": sum(zero_values) / len(zero_values) if zero_values else None,
            }
        )
    result = {
        "checkpoint": str(args.checkpoint),
        "model": model_name,
        "split": args.split,
        "record_count": len(records),
        "per_record": per_record,
        "model_mse": sum(all_prediction_errors) / len(all_prediction_errors) if all_prediction_errors else None,
        "zero_mse": sum(all_zero_errors) / len(all_zero_errors) if all_zero_errors else None,
        "model_below_zero": bool(all_prediction_errors and sum(all_prediction_errors) / len(all_prediction_errors) < sum(all_zero_errors) / len(all_zero_errors)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
