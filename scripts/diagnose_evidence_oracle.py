#!/usr/bin/env python3
"""Fit frozen linear oracles on FlyRot evidence without changing the model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from flyrot.data.tartanair import TartanAirWindowDataset, default_split
from flyrot.models.flyrot_v0 import FlyRotV0


def evenly_spaced_indices(length: int, limit: int) -> list[int]:
    if limit >= length:
        return list(range(length))
    if limit <= 1:
        return [0]
    return sorted({round(index * (length - 1) / (limit - 1)) for index in range(limit)})


def collect(
    model: FlyRotV0,
    dataset: TartanAirWindowDataset,
    indices: list[int],
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    loader = DataLoader(Subset(dataset, indices), batch_size=batch_size, shuffle=False, num_workers=0)
    features: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            frames = batch["frames"].to(device, non_blocking=True)
            on, off = model.photoreceptor(frames)
            energy, valid = model.direction_cells(on, off)
            evidence = model.rotation_evidence(energy, valid)
            state = model.accumulator(evidence)
            final_energy = energy[:, -1]
            final_evidence = evidence[:, -1]
            final_state = state[:, -1]
            feature = torch.cat(
                (
                    final_state,
                    final_evidence,
                    final_energy.mean(dim=(2, 3)),
                    final_energy.amax(dim=(2, 3)),
                ),
                dim=-1,
            )
            features.append(feature.cpu().numpy())
            targets.append(batch["target_rotation_vector"].numpy())
    return np.concatenate(features, axis=0), np.concatenate(targets, axis=0)


def fit_ridge(
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
    ridge: float,
) -> tuple[np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=0)
    scale = train_x.std(axis=0)
    scale[scale < 1e-8] = 1.0
    x_train = (train_x - mean) / scale
    x_validation = (validation_x - mean) / scale
    x_train = np.concatenate((np.ones((len(x_train), 1)), x_train), axis=1)
    x_validation = np.concatenate((np.ones((len(x_validation), 1)), x_validation), axis=1)
    regularizer = np.eye(x_train.shape[1], dtype=np.float64) * ridge
    regularizer[0, 0] = 0.0
    weights = np.linalg.solve(x_train.T @ x_train + regularizer, x_train.T @ train_y)
    return x_train @ weights, x_validation @ weights


def summarize(prediction: np.ndarray, target: np.ndarray) -> dict[str, float | bool]:
    model_mse = float(np.mean((prediction - target) ** 2))
    zero_mse = float(np.mean(target**2))
    return {
        "model_mse": model_mse,
        "zero_mse": zero_mse,
        "model_below_zero": model_mse < zero_mse,
        "target_norm_mean": float(np.linalg.norm(target, axis=1).mean()),
        "prediction_norm_mean": float(np.linalg.norm(prediction, axis=1).mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--window-length", type=int, default=3)
    parser.add_argument("--frame-gap", type=int, default=4)
    parser.add_argument("--max-train-samples", type=int, default=1024)
    parser.add_argument("--samples-per-validation-record", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--ridge", type=float, default=1e-2)
    parser.add_argument("--preload-workers", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = FlyRotV0().to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    with args.index.open(encoding="utf-8") as handle:
        records = json.load(handle)
    splits = default_split(records)

    train_dataset = TartanAirWindowDataset(
        splits["train"],
        window_length=args.window_length,
        frame_gap=args.frame_gap,
        image_size=args.image_size,
        preload_images=True,
        preload_workers=args.preload_workers,
    )
    train_indices = evenly_spaced_indices(len(train_dataset), min(args.max_train_samples, len(train_dataset)))
    train_x, train_y = collect(model, train_dataset, train_indices, device, args.batch_size)

    validation_features: list[np.ndarray] = []
    validation_targets: list[np.ndarray] = []
    per_record: list[dict] = []
    for record in splits["validation"]:
        dataset = TartanAirWindowDataset(
            [record],
            window_length=args.window_length,
            frame_gap=args.frame_gap,
            image_size=args.image_size,
            preload_images=True,
            preload_workers=args.preload_workers,
        )
        indices = evenly_spaced_indices(len(dataset), min(args.samples_per_validation_record, len(dataset)))
        feature, target = collect(model, dataset, indices, device, args.batch_size)
        validation_features.append(feature)
        validation_targets.append(target)
        per_record.append({"environment": record["environment"], "trajectory": record["trajectory"], "samples": len(target)})

    validation_x = np.concatenate(validation_features, axis=0)
    validation_y = np.concatenate(validation_targets, axis=0)
    train_prediction, prediction = fit_ridge(train_x, train_y, validation_x, args.ridge)
    ridge_sweep = {}
    for ridge in (1e-4, 1e-2, 1.0, 100.0):
        train_fit, validation_fit = fit_ridge(train_x, train_y, validation_x, ridge)
        ridge_sweep[str(ridge)] = {
            "train": summarize(train_fit, train_y),
            "validation": summarize(validation_fit, validation_y),
        }
    result = {
        "checkpoint": str(args.checkpoint),
        "frame_gap": args.frame_gap,
        "train_samples": len(train_y),
        "validation_samples": len(validation_y),
        "feature_count": train_x.shape[1],
        "ridge": args.ridge,
        "train_target_norm_mean": float(np.linalg.norm(train_y, axis=1).mean()),
        "train_fit": summarize(train_prediction, train_y),
        "validation": summarize(prediction, validation_y),
        "ridge_sweep": ridge_sweep,
        "per_record": per_record,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
