#!/usr/bin/env python3
"""Calibrate an input-dependent uncertainty head with rotation weights frozen."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from flyrot.data.tartanair import TartanAirWindowDataset, default_split
from flyrot.losses import heteroscedastic_nll
from flyrot.models.flyrot_v0 import FlyRotV0


def evenly_spaced_indices(length: int, limit: int) -> list[int]:
    if limit >= length:
        return list(range(length))
    if limit <= 1:
        return [0]
    return sorted({round(index * (length - 1) / (limit - 1)) for index in range(limit)})


def make_model(device: torch.device) -> FlyRotV0:
    return FlyRotV0(
        scales=(4, 8, 16),
        readout_scale=1.0,
        input_dependent_uncertainty=True,
    ).to(device)


def collect(
    model: FlyRotV0,
    datasets: list[TartanAirWindowDataset],
    device: torch.device,
    samples_per_record: int,
) -> dict:
    model.eval()
    errors: list[float] = []
    confidences: list[float] = []
    with torch.no_grad():
        for dataset in datasets:
            indices = evenly_spaced_indices(len(dataset), min(samples_per_record, len(dataset)))
            loader = DataLoader(Subset(dataset, indices), batch_size=32, shuffle=False, num_workers=0)
            for batch in loader:
                output = model(batch["frames"].to(device))
                prediction = output["rotation_vector"][:, -1]
                target = batch["target_rotation_vector"].to(device)
                errors.extend((prediction - target).square().mean(dim=1).cpu().tolist())
                confidences.extend(output["confidence"][:, -1, 0].cpu().tolist())
    errors_array = np.asarray(errors, dtype=np.float64)
    confidence_array = np.asarray(confidences, dtype=np.float64)
    order = np.argsort(confidence_array)
    bins = []
    for chunk in np.array_split(order, 5):
        if len(chunk):
            bins.append(
                {
                    "samples": int(len(chunk)),
                    "confidence": float(confidence_array[chunk].mean()),
                    "mse": float(errors_array[chunk].mean()),
                }
            )
    correlation = None
    if len(errors_array) > 1 and np.std(confidence_array) > 1e-12 and np.std(errors_array) > 1e-12:
        correlation = float(np.corrcoef(confidence_array, errors_array)[0, 1])
    return {
        "samples": int(len(errors_array)),
        "mse": float(errors_array.mean()) if len(errors_array) else None,
        "confidence_error_correlation": correlation,
        "confidence_bins_low_to_high": bins,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--max-train-samples", type=int, default=4096)
    parser.add_argument("--samples-per-record", type=int, default=512)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--window-length", type=int, default=3)
    parser.add_argument("--frame-gap", type=int, default=4)
    parser.add_argument("--target-direction", default="optical_image_motion")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--preload-workers", type=int, default=8)
    args = parser.parse_args()
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = make_model(device)
    model.load_state_dict(checkpoint["model"], strict=False)
    for name, parameter in model.named_parameters():
        parameter.requires_grad = name.startswith("uncertainty_")
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=1e-2)
    records = json.loads(args.index.read_text(encoding="utf-8"))
    splits = default_split(records)
    image_kwargs = {
        "window_length": args.window_length,
        "frame_gap": args.frame_gap,
        "image_size": args.image_size,
        "target_direction": args.target_direction,
        "preload_images": True,
        "preload_workers": args.preload_workers,
    }
    train_dataset = TartanAirWindowDataset(splits["train"], **image_kwargs)
    train_indices = evenly_spaced_indices(len(train_dataset), min(args.max_train_samples, len(train_dataset)))
    loader = DataLoader(Subset(train_dataset, train_indices), batch_size=32, shuffle=True, num_workers=0)
    iterator = iter(loader)
    model.train()
    for _ in range(args.steps):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        output = model(batch["frames"].to(device))
        prediction = output["rotation_vector"][:, -1].detach()
        target = batch["target_rotation_vector"].to(device)
        log_variance = output["log_variance"][:, -1]
        loss = heteroscedastic_nll(prediction, target, log_variance)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    validation_datasets = [TartanAirWindowDataset([record], **image_kwargs) for record in splits["validation"]]
    result = {
        "source_checkpoint": str(args.checkpoint),
        "steps": args.steps,
        "trainable_parameters": sum(parameter.numel() for parameter in trainable),
        "validation": collect(model, validation_datasets, device, args.samples_per_record),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    state = {
        "model": model.state_dict(),
        "config": {"model": "flyrot_motion_uncertainty", "source_checkpoint": str(args.checkpoint)},
        "calibration": result,
    }
    torch.save(state, args.output.with_suffix(".pt"))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
