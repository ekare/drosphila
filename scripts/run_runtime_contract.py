#!/usr/bin/env python3
"""Run capability-neutral CPU/CUDA runtime, serialization, and parity checks."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import time
from pathlib import Path

import torch

from flyrot.models.flyrot_v1 import FlyRotV1


def load_state(checkpoint: Path) -> dict[str, torch.Tensor]:
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    return state["model"] if "model" in state else state


def run_device(state: dict[str, torch.Tensor], device_name: str, frames_cpu: torch.Tensor, *, steps: int) -> tuple[dict[str, object], torch.Tensor]:
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        return {"status": "unavailable"}, torch.empty(0)
    model = FlyRotV1(scales=(4, 8, 16)).to(device)
    model.load_state_dict(state)
    model.eval()
    frames = frames_cpu.to(device)
    with torch.no_grad():
        for _ in range(2):
            model(frames)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        output = None
        for _ in range(steps):
            output = model(frames)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
    assert output is not None
    tensor = output["rotation_vector"].detach().cpu()
    finite = all(torch.isfinite(value).all().item() for value in output.values() if torch.is_tensor(value))
    result = {
        "status": "passed" if finite else "failed_nan_or_inf",
        "device": device_name,
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "steps": steps,
        "input_shape": [1, 3, 1, 128, 128],
        "latency_ms_per_step": elapsed * 1000.0 / steps,
        "peak_cuda_memory_mib": torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else 0.0,
        "finite_outputs": bool(finite),
    }
    return result, tensor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--devices", nargs="+", default=["cpu", "cuda:0", "cuda:1"])
    args = parser.parse_args()
    state = load_state(args.checkpoint)
    frames_cpu = torch.rand((1, 3, 1, 128, 128), generator=torch.Generator(device="cpu").manual_seed(args.seed))
    serialized = io.BytesIO()
    torch.save(state, serialized)
    restored = torch.load(io.BytesIO(serialized.getvalue()), map_location="cpu", weights_only=False)
    serialization_ok = set(state) == set(restored) and all(torch.equal(state[key], restored[key]) for key in state)
    device_results: dict[str, object] = {}
    outputs: dict[str, torch.Tensor] = {}
    for device in args.devices:
        result, tensor = run_device(state, device, frames_cpu, steps=args.steps)
        device_results[device] = result
        if tensor.numel():
            outputs[device] = tensor
    parity: dict[str, object] = {"status": "not_run"}
    if "cpu" in outputs and "cuda:1" in outputs:
        difference = (outputs["cpu"] - outputs["cuda:1"]).abs()
        parity = {"status": "passed" if float(difference.max()) < 2e-5 else "failed", "tolerance_max_abs": 2e-5, "max_abs": float(difference.max()), "mean_abs": float(difference.mean())}
    report = {
        "schema": "flyrot.runtime-contract.v1",
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "serialization": {"status": "passed" if serialization_ok else "failed", "bytes": len(serialized.getvalue())},
        "devices": device_results,
        "cpu_cuda_parity": parity,
        "claim_boundary": "runtime and serialization only; does not accept the learned motion field or downstream capabilities",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
