#!/usr/bin/env python3
"""Deterministic scale-response audit for native direction evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from flyrot.diagnostics import estimate_rotation_evidence, exact_rotational_flow
from flyrot.geometry.rotational_flow import CameraIntrinsics
from flyrot.synthetic import _direction_energy


def _json_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return _json_value(value.detach().cpu().tolist())
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def _texture_weight(height: int, width: int, seed: int, contrast: float) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    noise = torch.rand((1, 1, height, width), generator=generator)
    weight = 0.5 + contrast * (noise - 0.5)
    return weight.clamp_min(0.05)


def _region_mask(height: int, width: int, region: str) -> torch.Tensor:
    mask = torch.zeros((1, 1, height, width))
    if region == "full":
        mask.fill_(1.0)
    elif region == "center":
        mask[..., height // 4 : height - height // 4, width // 4 : width - width // 4] = 1.0
    elif region == "top_left":
        mask[..., : height // 2, : width // 2] = 1.0
    elif region == "edge":
        mask.fill_(1.0)
        mask[..., height // 5 : height - height // 5, width // 5 : width - width // 5] = 0.0
    else:
        raise ValueError(region)
    return mask


def _direction_accuracy(
    estimated: torch.Tensor,
    truth_flow: torch.Tensor,
    weight: torch.Tensor,
    intrinsics: CameraIntrinsics,
) -> float:
    estimated_flow = exact_rotational_flow(estimated, intrinsics=intrinsics)
    a = estimated_flow[0, 0]
    b = truth_flow[0, 0]
    cosine = (a * b).sum(dim=0) / (torch.linalg.vector_norm(a, dim=0) * torch.linalg.vector_norm(b, dim=0)).clamp_min(1e-8)
    valid = (torch.linalg.vector_norm(b, dim=0) > 1e-7) & (weight[0, 0] > 0)
    if not valid.any():
        return 0.0
    return float(cosine[valid].clamp(0, 1).mean())


def run(args: argparse.Namespace) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    height = width = args.image_size
    axes = {
        "rx": torch.tensor([1.0, 0.0, 0.0]),
        "ry": torch.tensor([0.0, 1.0, 0.0]),
        "rz": torch.tensor([0.0, 0.0, 1.0]),
    }
    intrinsics_cases = {
        "centered": CameraIntrinsics.centered(width, height),
        "shifted_principal": CameraIntrinsics(64.0, 51.0, 58.0, 69.0, width, height),
        "non_square_focal": CameraIntrinsics(74.0, 49.0, 63.5, 63.5, width, height),
    }
    for intrinsics_name, intrinsics in intrinsics_cases.items():
        for region in ("full", "center", "top_left", "edge"):
            region_mask = _region_mask(height, width, region)
            for texture_seed in (0, 1):
                for contrast in (0.25, 1.0):
                    texture = _texture_weight(height, width, texture_seed, contrast)
                    weight = texture * region_mask
                    for axis_name, axis in axes.items():
                        for sign in (-1.0, 1.0):
                            for angle in (0.005, 0.02, 0.08, 0.20):
                                rotation = (axis * (sign * angle)).view(1, 1, 3)
                                flow = exact_rotational_flow(rotation, intrinsics=intrinsics)
                                energy = _direction_energy(flow) * weight.view(1, 1, 1, height, width)
                                evidence = estimate_rotation_evidence(energy, torch.ones_like(energy), intrinsics)
                                per_scale = evidence["rotation_evidence"][0, 0]
                                total_response = energy.reshape(1, 1, 3, 8, height, width).sum(dim=(3, 4, 5))[0, 0]
                                flow_magnitude = torch.linalg.vector_norm(flow[0, 0], dim=0)
                                pixel_flow_magnitude = torch.sqrt(
                                    (flow[0, 0, 0] * intrinsics.fx) ** 2 + (flow[0, 0, 1] * intrinsics.fy) ** 2
                                )
                                spatial_weight = weight[0, 0]
                                normalized_displacement = float((flow_magnitude * spatial_weight).sum() / spatial_weight.sum().clamp_min(1e-8))
                                pixel_displacement = float((pixel_flow_magnitude * spatial_weight).sum() / spatial_weight.sum().clamp_min(1e-8))
                                for scale_index in range(3):
                                    estimate = per_scale[scale_index]
                                    rows.append(
                                        {
                                            "axis": axis_name,
                                            "sign": sign,
                                            "angle_rad": angle,
                                            "image_region": region,
                                            "texture_seed": texture_seed,
                                            "contrast": contrast,
                                            "intrinsics": intrinsics_name,
                                            "frame_gap": args.frame_gap,
                                            "scale_index": scale_index,
                                            "total_response": float(total_response[scale_index]),
                                            "evidence_norm": float(torch.linalg.vector_norm(estimate)),
                                            "preferred_displacement": float(torch.linalg.vector_norm(estimate)),
                                            "normalized_displacement": normalized_displacement,
                                            "pixel_displacement": pixel_displacement,
                                            "direction_accuracy": _direction_accuracy(
                                                estimate.view(1, 1, 3), flow, spatial_weight, intrinsics
                                            ),
                                            "evidence_valid": bool(evidence["evidence_valid"][0, 0, scale_index]),
                                        }
                                    )
    report = {
        "experiment": "direction-scale-response-v0.3.0",
        "image_size": args.image_size,
        "rows": rows,
        "interpretation": "Direction-cell evidence is directional; response magnitude was not calibrated as physical displacement.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(_json_value(report), indent=2) + "\n", encoding="utf-8")
    artifact_path = args.artifact_dir / "scale_calibration_rows.json"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(json.dumps(_json_value(report), indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("reports/scale_calibration_v0.3.0.json"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts/scale_calibration"))
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--frame-gap", type=int, default=4)
    args = parser.parse_args()
    report = run(args)
    print(json.dumps({"rows": len(report["rows"]), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
