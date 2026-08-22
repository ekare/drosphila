#!/usr/bin/env python3
"""Measure RGB-only uncertainty proxies without changing checkpoint weights."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from flyrot.data.tartanair import TartanAirWindowDataset, default_split
from flyrot.metrics import per_sample_geodesic_rad
from flyrot.models.flyrot_v0 import FlyRotV0


def indices(length: int, limit: int) -> list[int]:
    if limit >= length:
        return list(range(length))
    return sorted({round(i * (length - 1) / max(1, limit - 1)) for i in range(limit)})


def summarize(error: torch.Tensor, value: torch.Tensor, high_is_confidence: bool) -> dict:
    value = value.reshape(-1)
    error = error.reshape(-1)
    if not high_is_confidence:
        value = -value
    result = {
        "mean": float(value.mean()),
        "std": float(value.std(unbiased=False)),
        "error_correlation": None,
        "selective_risk_deg": [],
    }
    if value.std(unbiased=False) > 1e-8:
        result["error_correlation"] = float(torch.corrcoef(torch.stack((value, error)))[0, 1])
    order = torch.argsort(value, descending=True)
    for coverage in (0.25, 0.5, 0.75, 1.0):
        count = max(1, int(np.ceil(len(order) * coverage)))
        result["selective_risk_deg"].append(
            {
                "coverage": coverage,
                "samples": count,
                "risk": float(torch.rad2deg(error[order[:count]]).mean()),
            }
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="validation")
    parser.add_argument("--samples-per-record", type=int, default=10000)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--frame-gap", type=int, default=4)
    parser.add_argument("--target-direction", default="optical_image_motion")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--preload-workers", type=int, default=8)
    args = parser.parse_args()
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = FlyRotV0(scales=(4, 8, 16), readout_scale=1.0, magnitude_confidence=True).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    records = default_split(json.loads(args.index.read_text(encoding="utf-8")))[args.split]
    errors = []
    proxies: dict[str, list[torch.Tensor]] = {
        "motion_coherence": [],
        "pair_composition_consistency": [],
        "prediction_magnitude": [],
    }
    with torch.no_grad():
        for record in records:
            dataset = TartanAirWindowDataset(
                [record],
                window_length=3,
                frame_gap=args.frame_gap,
                image_size=args.image_size,
                target_direction=args.target_direction,
                preload_images=True,
                preload_workers=args.preload_workers,
            )
            loader = DataLoader(
                Subset(dataset, indices(len(dataset), min(args.samples_per_record, len(dataset)))),
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=0,
            )
            for batch in loader:
                frames = batch["frames"].to(device)
                endpoint = model(frames)
                prediction = endpoint["rotation_vector"][:, -1]
                target = batch["target_rotation_vector"].to(device)
                errors.append(per_sample_geodesic_rad(prediction, target).cpu())
                coherence = model.rotation_evidence.rotational_coherence(
                    endpoint["direction_energy"], endpoint["valid_mask"]
                )[:, -1, 0]
                pair_a = model(frames[:, :2])["rotation_vector"][:, -1]
                pair_b = model(frames[:, 1:])["rotation_vector"][:, -1]
                composition_error = torch.linalg.vector_norm(prediction - pair_a - pair_b, dim=-1)
                proxies["motion_coherence"].append(coherence.cpu())
                proxies["pair_composition_consistency"].append(composition_error.cpu())
                proxies["prediction_magnitude"].append(torch.linalg.vector_norm(prediction, dim=-1).cpu())
    error = torch.cat(errors)
    result = {
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "samples": int(len(error)),
        "mean_error_deg": float(torch.rad2deg(error).mean()),
        "proxies": {},
    }
    for name, values in proxies.items():
        high_is_confidence = name not in {"pair_composition_consistency"}
        result["proxies"][name] = summarize(error, torch.cat(values), high_is_confidence)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
