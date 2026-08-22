#!/usr/bin/env python3
"""Checkpointed FlyRot training with trajectory-disjoint validation."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from flyrot.data.tartanair import TartanAirWindowDataset, default_split
from flyrot.losses import flyrot_loss
from flyrot.models.flyrot_v0 import FlyRotV0
from flyrot.models.tiny_conv_baseline import TinyConvBaseline


def build_model(model_name: str) -> torch.nn.Module:
    if model_name == "flyrot":
        return FlyRotV0()
    if model_name == "flyrot_gated":
        return FlyRotV0(confidence_gated=True, readout_scale=8.0)
    if model_name == "flyrot_magnitude_gated":
        return FlyRotV0(confidence_gated=True, readout_scale=8.0, magnitude_aware=True)
    if model_name == "flyrot_appearance_gated":
        return FlyRotV0(confidence_gated=True, readout_scale=8.0, appearance_normalized=True)
    if model_name == "flyrot_scaled_gated":
        return FlyRotV0(scales=(2, 4, 8), confidence_gated=True, readout_scale=8.0)
    if model_name == "flyrot_motion_scaled_gated":
        return FlyRotV0(scales=(4, 8, 16), confidence_gated=True, readout_scale=8.0)
    if model_name == "flyrot_scale_separated_gated":
        return FlyRotV0(
            scales=(4, 8, 16),
            confidence_gated=True,
            readout_scale=8.0,
            scale_separated=True,
            magnitude_aware=True,
        )
    if model_name == "flyrot_lstsq_gated":
        return FlyRotV0(scales=(4, 8, 16), confidence_gated=True, readout_scale=8.0, least_squares_basis=True)
    if model_name == "flyrot_motion_direct":
        return FlyRotV0(scales=(4, 8, 16), readout_scale=1.0, magnitude_confidence=True)
    if model_name == "flyrot_motion_consistency":
        return FlyRotV0(scales=(4, 8, 16), readout_scale=1.0, motion_consistency_gate=True)
    if model_name == "flyrot_motion_uncertainty":
        return FlyRotV0(scales=(4, 8, 16), readout_scale=1.0, input_dependent_uncertainty=True)
    if model_name == "flyrot_stretched_gated":
        return FlyRotV0(confidence_gated=True, readout_scale=8.0, focal_y_over_x=4.0 / 3.0)
    if model_name == "tiny_conv_32":
        return TinyConvBaseline(channels=32)
    raise ValueError(f"unknown model: {model_name}")


def evenly_spaced_indices(length: int, limit: int) -> list[int]:
    if limit >= length:
        return list(range(length))
    if limit <= 1:
        return [0]
    return sorted({round(index * (length - 1) / (limit - 1)) for index in range(limit)})


def evaluate(model: torch.nn.Module, datasets: list[TartanAirWindowDataset], device: torch.device, limit: int) -> dict:
    model.eval()
    model_errors: list[float] = []
    zero_errors: list[float] = []
    geodesic_errors: list[float] = []
    per_record = []
    with torch.no_grad():
        for dataset in datasets:
            indices = evenly_spaced_indices(len(dataset), min(limit, len(dataset)))
            loader = DataLoader(
                Subset(dataset, indices),
                batch_size=32,
                shuffle=False,
                num_workers=0,
                pin_memory=device.type == "cuda",
            )
            record_model: list[float] = []
            record_zero: list[float] = []
            record_geo: list[float] = []
            for batch in loader:
                frames = batch["frames"].to(device)
                target = batch["target_rotation_vector"].to(device)
                output = model(frames)
                prediction = output["rotation_vector"][:, -1]
                record_model.extend((prediction - target).square().mean(dim=1).cpu().tolist())
                record_zero.extend(target.square().mean(dim=1).cpu().tolist())
                record_geo.extend(
                    flyrot_loss(prediction, target, output["log_variance"][:, -1])["geodesic"]
                    .detach()
                    .cpu()
                    .reshape(-1)
                    .tolist()
                )
            model_errors.extend(record_model)
            zero_errors.extend(record_zero)
            geodesic_errors.extend(record_geo)
            if record_model:
                metadata = dataset.records[0]
                per_record.append(
                    {
                        "environment": metadata["environment"],
                        "trajectory": metadata["trajectory"],
                        "samples": len(record_model),
                        "model_mse": float(np.mean(record_model)),
                        "zero_mse": float(np.mean(record_zero)),
                    }
                )
    model.eval()
    return {
        "model_mse": float(np.mean(model_errors)) if model_errors else None,
        "zero_mse": float(np.mean(zero_errors)) if zero_errors else None,
        "geodesic_rad": float(np.mean(geodesic_errors)) if geodesic_errors else None,
        "model_below_zero": bool(model_errors and np.mean(model_errors) < np.mean(zero_errors)),
        "per_record": per_record,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, default=Path("artifacts/tartanair2_index.json"))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--max-train-samples", type=int, default=1024)
    parser.add_argument("--max-val-samples", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--image-height", type=int)
    parser.add_argument("--window-length", type=int, default=3)
    parser.add_argument("--frame-gap", type=int, default=4)
    parser.add_argument(
        "--target-direction",
        choices=("camera_relative", "image_motion", "optical_relative", "optical_image_motion"),
        default="camera_relative",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default=None, help="torch device; defaults to CUDA when available, otherwise CPU")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--preload-images", action="store_true")
    parser.add_argument("--preload-workers", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--geodesic-weight", type=float, default=1.0)
    parser.add_argument("--huber-weight", type=float, default=0.25)
    parser.add_argument("--uncertainty-weight", type=float, default=0.1)
    parser.add_argument("--step-loss-weight", type=float, default=0.0)
    parser.add_argument(
        "--model",
        choices=(
            "flyrot",
            "flyrot_gated",
            "flyrot_magnitude_gated",
            "flyrot_appearance_gated",
            "flyrot_scaled_gated",
            "flyrot_motion_scaled_gated",
            "flyrot_scale_separated_gated",
            "flyrot_lstsq_gated",
            "flyrot_motion_direct",
            "flyrot_motion_consistency",
            "flyrot_motion_uncertainty",
            "flyrot_stretched_gated",
            "tiny_conv_32",
        ),
        default="flyrot",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    image_size = (args.image_size, args.image_height) if args.image_height else args.image_size

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    with args.index.open(encoding="utf-8") as handle:
        records = json.load(handle)
    splits = default_split(records)
    train_dataset = TartanAirWindowDataset(
        splits["train"],
        window_length=args.window_length,
        frame_gap=args.frame_gap,
        image_size=image_size,
        preload_images=args.preload_images,
        preload_workers=args.preload_workers,
        target_direction=args.target_direction,
    )
    train_indices = evenly_spaced_indices(len(train_dataset), min(args.max_train_samples, len(train_dataset)))
    loader_options = {
        "batch_size": args.batch_size,
        "shuffle": True,
        "num_workers": args.num_workers,
        "pin_memory": True,
    }
    if args.num_workers > 0:
        loader_options.update({"persistent_workers": True, "prefetch_factor": 2})
    train_loader = DataLoader(Subset(train_dataset, train_indices), **loader_options)
    validation_datasets = [
        TartanAirWindowDataset(
            [record],
            window_length=args.window_length,
            frame_gap=args.frame_gap,
            image_size=image_size,
            preload_images=args.preload_images,
            preload_workers=args.preload_workers,
            target_direction=args.target_direction,
        )
        for record in splits["validation"]
    ]
    requested_device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested_device)
    model = build_model(args.model).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    start_step = 0
    best_val = float("inf")
    if args.resume:
        state = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_step = int(state["step"])
        best_val = float(state.get("best_val_mse", best_val))

    event_path = args.run_dir / "events.jsonl"
    metrics_path = args.run_dir / "metrics.json"
    latest_path = args.run_dir / "latest.pt"
    best_path = args.run_dir / "best.pt"
    parameter_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    latest_metrics = None
    iterator = iter(train_loader)
    for step in range(start_step, args.steps):
        model.train()
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            batch = next(iterator)
        frames = batch["frames"].to(device)
        target = batch["target_rotation_vector"].to(device)
        output = model(frames)
        prediction = output["rotation_vector"][:, -1]
        components = flyrot_loss(
            prediction,
            target,
            output["log_variance"][:, -1],
            geodesic_weight=args.geodesic_weight,
            huber_weight=args.huber_weight,
            uncertainty_weight=args.uncertainty_weight,
        )
        if args.step_loss_weight:
            step_target = batch["target_rotation_vectors"].to(device)
            step_components = flyrot_loss(
                output["rotation_vector"].reshape(-1, 3),
                step_target.reshape(-1, 3),
                output["log_variance"].reshape(-1, 3),
                geodesic_weight=args.geodesic_weight,
                huber_weight=args.huber_weight,
                uncertainty_weight=args.uncertainty_weight,
            )
            components["total"] = components["total"] + args.step_loss_weight * step_components["total"]
        optimizer.zero_grad(set_to_none=True)
        components["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        completed_step = step + 1
        if completed_step % args.eval_interval == 0 or completed_step == args.steps:
            validation = evaluate(model, validation_datasets, device, args.max_val_samples)
            latest_metrics = {
                "step": completed_step,
                "train_loss": float(components["total"].detach().cpu()),
                "train_geodesic_rad": float(components["geodesic"].detach().cpu()),
                "train_rotation_huber": float(components["rotation_huber"].detach().cpu()),
                "train_uncertainty_nll": float(components["uncertainty_nll"].detach().cpu()),
                "validation": validation,
                "parameter_count": parameter_count,
                "device": torch.cuda.get_device_name(device) if device.type == "cuda" else str(device),
                "torch": torch.__version__,
                "cuda_runtime": torch.version.cuda,
            }
            with event_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(latest_metrics) + "\n")
            state = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": completed_step,
                "best_val_mse": best_val,
                "config": vars(args),
            }
            torch.save(state, latest_path)
            if validation["model_mse"] is not None and validation["model_mse"] < best_val:
                best_val = validation["model_mse"]
                state["best_val_mse"] = best_val
                torch.save(state, best_path)
            metrics_path.write_text(json.dumps(latest_metrics, indent=2) + "\n", encoding="utf-8")
            print(json.dumps(latest_metrics), flush=True)
    if latest_metrics is None:
        latest_metrics = {"step": start_step, "parameter_count": parameter_count}
        metrics_path.write_text(json.dumps(latest_metrics, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
