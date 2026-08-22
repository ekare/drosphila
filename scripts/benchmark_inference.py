#!/usr/bin/env python3
"""Benchmark one FlyRot checkpoint on the selected CUDA device."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from flyrot.models.flyrot_v0 import FlyRotV0
from flyrot.models.flyrot_v1 import FlyRotV1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--image-size", type=int, default=128)
    args = parser.parse_args()
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model_name = checkpoint.get("config", {}).get("model", "flyrot_motion_direct")
    if model_name == "flyrot_v1":
        model = FlyRotV1(scales=(4, 8, 16)).to(device)
    else:
        model = FlyRotV0(scales=(4, 8, 16), readout_scale=1.0, magnitude_confidence=model_name == "flyrot_motion_direct").to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    frames = torch.rand(args.batch_size, 3, 1, args.image_size, args.image_size, device=device)
    with torch.no_grad():
        for _ in range(10):
            model(frames)
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        for _ in range(args.steps):
            model(frames)
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start
    result = {
        "checkpoint": str(args.checkpoint),
        "device": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "batch_size": args.batch_size,
        "steps": args.steps,
        "window_length": 3,
        "image_size": args.image_size,
        "total_seconds": elapsed,
        "batch_latency_ms": 1000.0 * elapsed / args.steps,
        "sample_latency_ms": 1000.0 * elapsed / (args.steps * args.batch_size),
        "peak_cuda_memory_mib": torch.cuda.max_memory_allocated(device) / (1024**2),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
