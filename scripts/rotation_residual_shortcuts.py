#!/usr/bin/env python3
"""Run deterministic shortcut experiments against the residual diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import torch

from flyrot.diagnostics import build_rotation_residual_diagnostics
from flyrot.synthetic import make_synthetic_case

sys.path.insert(0, str(Path(__file__).parent))
from rotation_residual import _json_value, _last, save_panel  # noqa: E402


def _run(model: torch.nn.Module, frames: torch.Tensor, device: torch.device) -> dict[str, torch.Tensor]:
    with torch.no_grad():
        output = model(frames.unsqueeze(0).to(device), diagnostics=True)
    return {key: value.detach().cpu() for key, value in output.items() if isinstance(value, torch.Tensor)}


def _summary(result: dict[str, torch.Tensor], reference: torch.Tensor | None = None) -> dict[str, Any]:
    prediction = _last(result, "predicted_rotation")
    summary: dict[str, Any] = {
        "prediction": prediction,
        "prediction_norm": torch.linalg.vector_norm(prediction),
        "residual_ratio": _last(result, "residual_ratio"),
        "spatial_support": _last(result, "spatial_support"),
        "scale_agreement": _last(result, "scale_agreement"),
        "temporal_agreement": _last(result, "temporal_agreement"),
        "on_off_agreement": _last(result, "on_off_agreement"),
        "global_confidence": _last(result, "global_confidence"),
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
    variants = {
        "original": case.frames,
        "reversed": case.frames.flip(0),
        "repeated_first": case.frames[0:1].expand_as(case.frames).clone(),
        "brightness_half": (case.frames * 0.5).clamp(0, 1),
        "brightness_double": (case.frames * 2.0).clamp(0, 1),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {}
    original_result = _run(model, variants["original"], args.device)
    reference = _last(original_result, "predicted_rotation")
    for name, frames in variants.items():
        result = original_result if name == "original" else _run(model, frames, args.device)
        results[name] = _summary(result, reference)
        save_panel(
            frames,
            result,
            args.output / f"{name}.png",
            title=f"Shortcut experiment: {name}",
            ground_truth=case.pair_rotations[-1],
        )
    results["oracle_reference"] = _summary(
        build_rotation_residual_diagnostics(
            case.pair_rotations.unsqueeze(0),
            case.oracle_energy,
            torch.ones_like(case.oracle_energy),
            on_energy=case.oracle_on_energy,
            off_energy=case.oracle_off_energy,
        ).as_dict()
    )
    output_path = args.output / "metrics.json"
    output_path.write_text(json.dumps(_json_value(results), indent=2) + "\n", encoding="utf-8")
    print(json.dumps(_json_value(results), indent=2))


if __name__ == "__main__":
    main()
