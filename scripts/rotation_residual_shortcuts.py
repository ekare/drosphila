#!/usr/bin/env python3
"""Run deterministic shortcut and diagnostic-convention experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import torch

from flyrot.diagnostics import build_rotation_residual_diagnostics
from flyrot.geometry.rotational_flow import CameraIntrinsics
from flyrot.synthetic import make_synthetic_case

sys.path.insert(0, str(Path(__file__).parent))
from rotation_residual import _json_value, _last, save_panel  # noqa: E402


def _run(model: torch.nn.Module, frames: torch.Tensor, device: torch.device) -> dict[str, torch.Tensor]:
    with torch.no_grad():
        output = model(frames.unsqueeze(0).to(device), diagnostics=True)
    return {key: value.detach().cpu() for key, value in output.items() if isinstance(value, torch.Tensor)}


def _luminance_preserving_hue(frames: torch.Tensor) -> torch.Tensor:
    luminance = 0.299 * frames[:, 0:1] + 0.587 * frames[:, 1:2] + 0.114 * frames[:, 2:3]
    # Add a deterministic chroma direction orthogonal to the luminance row.
    chroma = torch.stack((frames[:, 1] - luminance[:, 0], frames[:, 2] - luminance[:, 0], frames[:, 0] - luminance[:, 0]), dim=1)
    return (luminance + 0.45 * chroma).clamp(0, 1)


def _edges_only(frames: torch.Tensor) -> torch.Tensor:
    gray = frames.mean(dim=1, keepdim=True)
    dx = gray[..., :, 1:] - gray[..., :, :-1]
    dy = gray[..., 1:, :] - gray[..., :-1, :]
    result = torch.zeros_like(gray)
    result[..., :, 1:] += dx.abs()
    result[..., 1:, :] += dy.abs()
    result = result / result.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
    return result.repeat(1, 3, 1, 1)


def _center_only(frames: torch.Tensor) -> torch.Tensor:
    result = torch.full_like(frames, 0.5)
    height, width = frames.shape[-2:]
    top, bottom = height // 4, height - height // 4
    left, right = width // 4, width - width // 4
    result[..., top:bottom, left:right] = frames[..., top:bottom, left:right]
    return result


def _moving_patch(frames: torch.Tensor) -> torch.Tensor:
    result = frames.clone()
    height, width = frames.shape[-2:]
    size = max(6, min(height, width) // 8)
    for step in range(len(frames)):
        left = min(width - size, width // 4 + step * max(2, width // 32))
        top = height // 3
        result[step, :, top : top + size, left : left + size] = torch.tensor(
            [0.95, 0.15, 0.08], dtype=frames.dtype, device=frames.device
        ).view(3, 1, 1)
    return result


def _diagnostic_variant(
    result: dict[str, torch.Tensor],
    rotation: torch.Tensor,
    *,
    intrinsics: CameraIntrinsics | None = None,
) -> dict[str, torch.Tensor]:
    energy = result["direction_energy"]
    valid = result["valid_mask"]
    on = result["on_motion_energy"]
    off = result["off_motion_energy"]
    diag = build_rotation_residual_diagnostics(
        rotation,
        energy,
        valid,
        on_energy=on,
        off_energy=off,
        intrinsics=intrinsics,
    ).as_dict()
    merged = dict(result)
    merged.update(diag)
    return merged


def _score(result: dict[str, torch.Tensor], score_key: str, valid_key: str) -> float | None:
    score = _last(result, score_key)
    valid = _last(result, valid_key).bool()
    if not bool(valid.item()) or not torch.isfinite(score).all():
        return None
    return float(score.item())


def _summary(result: dict[str, torch.Tensor], reference: torch.Tensor | None = None) -> dict[str, Any]:
    prediction = _last(result, "predicted_rotation")
    summary: dict[str, Any] = {
        "prediction": prediction,
        "prediction_norm": torch.linalg.vector_norm(prediction),
        "directional_residual_pred": _last(result, "directional_residual_ratio"),
        "directional_fit_score": _last(result, "directional_fit_score"),
        "spatial_support_score": _last(result, "spatial_support_score"),
        "scale_agreement_score": _score(result, "scale_agreement_score", "scale_agreement_valid"),
        "temporal_agreement_score": _score(result, "temporal_agreement_score", "temporal_agreement_valid"),
        "on_off_agreement_score": _score(result, "on_off_agreement_score", "on_off_agreement_valid"),
        "on_off_balance_score": _last(result, "on_off_balance_score"),
        "global_reliability": _last(result, "global_reliability"),
        "global_reliability_valid": _last(result, "global_reliability_valid"),
        "axis_confidence": _last(result, "axis_confidence"),
        "axis_variance": _last(result, "axis_variance"),
        "observability_condition": _last(result, "observability_condition"),
    }
    if "old_confidence" in result:
        summary["old_confidence"] = _last(result, "old_confidence")
    if reference is not None:
        summary["prediction_delta_norm"] = torch.linalg.vector_norm(prediction - reference)
        summary["prediction_cosine_to_original"] = torch.nn.functional.cosine_similarity(
            prediction.unsqueeze(0), reference.unsqueeze(0), dim=-1
        )[0]
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("models/flyrot_v0_best.pt"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/rotation_residual_shortcuts"))
    parser.add_argument("--device", default=None)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    args.device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))

    sys.path.insert(0, str(Path(__file__).parent))
    from rotation_residual import _model_from_checkpoint  # noqa: E402

    model = _model_from_checkpoint(args.checkpoint, args.device)
    case = make_synthetic_case("rotation", args.image_size, args.image_size, args.seed)
    frames = case.frames
    height, width = frames.shape[-2:]
    centered = CameraIntrinsics.centered(width, height)
    variants: dict[str, tuple[torch.Tensor, str]] = {
        "original": (frames, "model"),
        "time_reversed": (frames.flip(0), "model"),
        "time_shuffled": (frames[[0, 2, 1]], "model"),
        "same_frame_repeated": (frames[0:1].expand_as(frames).clone(), "model"),
        "brightness_x0.5": ((frames * 0.5).clamp(0, 1), "model"),
        "brightness_x2": ((frames * 2.0).clamp(0, 1), "model"),
        "gamma_increase": (frames.clamp_min(1e-4).pow(1.5), "model"),
        "gamma_decrease": (frames.clamp_min(1e-4).pow(0.7), "model"),
        "low_texture": ((0.5 + 0.05 * (frames - 0.5)).clamp(0, 1), "model"),
        "center_only": (_center_only(frames), "model"),
        "edges_only": (_edges_only(frames), "model"),
        "small_independent_moving_patch": (_moving_patch(frames), "model"),
        "hue_changed_luminance_preserved": (_luminance_preserving_hue(frames), "model"),
        "luminance_structure_damaged": (frames.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1), "model"),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {}
    original_result = _run(model, frames, args.device)
    reference = _last(original_result, "predicted_rotation")
    for name, (variant_frames, _) in variants.items():
        result = original_result if name == "original" else _run(model, variant_frames, args.device)
        results[name] = _summary(result, reference)
        save_panel(
            variant_frames,
            result,
            args.output / f"{name}.png",
            title=f"Shortcut experiment: {name}",
            ground_truth=case.pair_rotations[-1],
        )

    wrong_intrinsics = centered.cropped(width // 10, height // 12, width - width // 5, height - height // 6).resized(width, height)
    diagnostic_rotation = original_result["predicted_rotation"]
    results["wrong_intrinsics"] = _summary(
        _diagnostic_variant(original_result, diagnostic_rotation, intrinsics=CameraIntrinsics.centered(width, height, 0.7)),
        reference,
    )
    results["shifted_principal_point"] = _summary(
        _diagnostic_variant(
            original_result,
            diagnostic_rotation,
            intrinsics=CameraIntrinsics(
                fx=centered.fx,
                fy=centered.fy,
                cx=centered.cx + width * 0.1,
                cy=centered.cy - height * 0.07,
                width=width,
                height=height,
            ),
        ),
        reference,
    )
    results["wrong_intrinsics"]["intrinsics_transform"] = wrong_intrinsics.as_dict()
    results["wrong_rotation_sign"] = _summary(_diagnostic_variant(original_result, -diagnostic_rotation), reference)
    results["rotation_axes_permuted"] = _summary(
        _diagnostic_variant(original_result, diagnostic_rotation[..., [1, 2, 0]]), reference
    )
    oracle_result = build_rotation_residual_diagnostics(
        case.pair_rotations.unsqueeze(0),
        case.oracle_energy,
        torch.ones_like(case.oracle_energy),
        on_energy=case.oracle_on_energy,
        off_energy=case.oracle_off_energy,
    ).as_dict()
    results["oracle_reference"] = _summary(oracle_result)
    output_path = args.output / "metrics.json"
    output_path.write_text(json.dumps(_json_value(results), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(_json_value(results), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
