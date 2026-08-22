#!/usr/bin/env python3
"""Produce full rotation and selective-risk metrics for a checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).parent))
from train import build_model, evenly_spaced_indices  # noqa: E402

from flyrot.data.tartanair import TartanAirWindowDataset, default_split  # noqa: E402
from flyrot.metrics import rotation_metrics  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--samples-per-record", type=int, default=10000)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--window-length", type=int, default=3)
    parser.add_argument("--frame-gap", type=int, default=4)
    parser.add_argument("--target-direction", default="optical_image_motion")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--preload-images", action="store_true")
    parser.add_argument("--preload-workers", type=int, default=8)
    args = parser.parse_args()
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = build_model(checkpoint.get("config", {}).get("model", "flyrot")).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    records = default_split(json.loads(args.index.read_text(encoding="utf-8")))[args.split]
    per_record = []
    all_prediction = []
    all_target = []
    all_confidence = []
    all_coherence = []
    with torch.no_grad():
        for record in records:
            dataset = TartanAirWindowDataset(
                [record],
                window_length=args.window_length,
                frame_gap=args.frame_gap,
                image_size=args.image_size,
                target_direction=args.target_direction,
                preload_images=args.preload_images,
                preload_workers=args.preload_workers,
            )
            indices = evenly_spaced_indices(len(dataset), min(args.samples_per_record, len(dataset)))
            loader = DataLoader(Subset(dataset, indices), batch_size=args.batch_size, shuffle=False, num_workers=0)
            predictions = []
            targets = []
            confidences = []
            coherences = []
            for batch in loader:
                output = model(batch["frames"].to(device))
                predictions.append(output["rotation_vector"][:, -1].cpu())
                targets.append(batch["target_rotation_vector"])
                confidences.append(output["confidence"][:, -1, 0].cpu())
                coherences.append(
                    model.rotation_evidence.rotational_coherence(output["direction_energy"], output["valid_mask"])
                    [:, -1, 0]
                    .cpu()
                )
            prediction = torch.cat(predictions)
            target = torch.cat(targets)
            confidence = torch.cat(confidences)
            coherence = torch.cat(coherences)
            metrics = rotation_metrics(prediction, target, confidence)
            metrics["motion_coherence"] = rotation_metrics(prediction, target, coherence)
            metrics.update({"environment": record["environment"], "trajectory": record["trajectory"]})
            per_record.append(metrics)
            all_prediction.append(prediction)
            all_target.append(target)
            all_confidence.append(confidence)
            all_coherence.append(coherence)
    prediction = torch.cat(all_prediction)
    target = torch.cat(all_target)
    confidence = torch.cat(all_confidence)
    coherence = torch.cat(all_coherence)
    result = {
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "target_direction": args.target_direction,
        "window_length": args.window_length,
        "frame_gap": args.frame_gap,
        "model_parameter_count": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "metrics": rotation_metrics(prediction, target, confidence),
        "motion_coherence_metrics": rotation_metrics(prediction, target, coherence),
        "per_record": per_record,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["metrics"], indent=2))


if __name__ == "__main__":
    main()
