#!/usr/bin/env python3
"""Small real-data FlyRot smoke/overfit run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from flyrot.data.tartanair import TartanAirWindowDataset, default_split
from flyrot.losses import flyrot_loss
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
    parser.add_argument("--index", type=Path, default=Path("artifacts/tartanair2_index.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/smoke_metrics.json"))
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/smoke_checkpoint.pt"))
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--window-length", type=int, default=3)
    parser.add_argument("--frame-gap", type=int, default=1)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--all-train", action="store_true")
    parser.add_argument("--model", choices=("flyrot", "tiny_conv"), default="flyrot")
    args = parser.parse_args()

    with args.index.open(encoding="utf-8") as handle:
        records = json.load(handle)
    split = default_split(records)
    train_records = split["train"] if args.all_train else [split["train"][0]]
    dataset = TartanAirWindowDataset(
        train_records,
        window_length=args.window_length,
        frame_gap=args.frame_gap,
        image_size=args.image_size,
    )
    subset = Subset(dataset, evenly_spaced_indices(len(dataset), min(args.samples, len(dataset))))
    loader = DataLoader(subset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    device = torch.device(args.device)
    model = (FlyRotV0() if args.model == "flyrot" else TinyConvBaseline()).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    parameter_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)

    first_loss = None
    last_loss = None
    first_rotation_mse = None
    last_rotation_mse = None
    history: list[float] = []
    iterator = iter(loader)
    for step in range(args.steps):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        frames = batch["frames"].to(device)
        target = batch["target_rotation_vector"].to(device)
        output = model(frames)
        prediction = output["rotation_vector"][:, -1]
        rotation_mse = (prediction - target).square().mean()
        components = flyrot_loss(prediction, target, output["log_variance"][:, -1])
        loss = components["total"]
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        value = float(loss.detach().cpu())
        first_loss = value if first_loss is None else first_loss
        last_loss = value
        rotation_value = float(rotation_mse.detach().cpu())
        first_rotation_mse = rotation_value if first_rotation_mse is None else first_rotation_mse
        last_rotation_mse = rotation_value
        if step == 0 or (step + 1) % max(1, args.steps // 10) == 0:
            history.append(value)

    with torch.no_grad():
        target_values = torch.cat([batch["target_rotation_vector"] for batch in loader], dim=0)
        zero_baseline = float(target_values.square().mean())
    torch.save({"model": model.state_dict(), "step": args.steps, "config": vars(args)}, args.checkpoint)
    result = {
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else str(device),
        "model": args.model,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "dataset_records": dataset.records,
        "sample_count": len(subset),
        "steps": args.steps,
        "batch_size": args.batch_size,
        "window_length": args.window_length,
        "frame_gap": args.frame_gap,
        "image_size": args.image_size,
        "parameter_count": parameter_count,
        "zero_baseline_mse": zero_baseline,
        "first_loss": first_loss,
        "last_loss": last_loss,
        "first_rotation_mse": first_rotation_mse,
        "last_rotation_mse": last_rotation_mse,
        "loss_history": history,
        "loss_decreased": bool(last_loss < first_loss),
        "rotation_mse_below_zero_baseline": bool(last_rotation_mse < zero_baseline),
        "peak_vram_mib": float(torch.cuda.max_memory_allocated(device) / 2**20) if device.type == "cuda" else 0.0,
        "checkpoint": str(args.checkpoint),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
