#!/usr/bin/env python3
"""Generate rotation-residual diagnostics for synthetic or real RGB windows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from flyrot.data.tartanair import TartanAirWindowDataset, default_split
from flyrot.diagnostics import build_rotation_residual_diagnostics
from flyrot.metrics import rotation_metrics
from flyrot.models.flyrot_v0 import FlyRotV0
from flyrot.synthetic import SyntheticCase, make_synthetic_case


def _json_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return float(value.detach().cpu().item())
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def _last(result: dict[str, torch.Tensor], key: str) -> torch.Tensor:
    value = result[key]
    return value[0, -1].detach().cpu()


def _image(frames: torch.Tensor, index: int) -> np.ndarray:
    return frames[index].detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()


def _map(result: dict[str, torch.Tensor], key: str) -> np.ndarray:
    value = _last(result, key)
    if value.ndim == 3:
        value = value.sum(dim=0)
    return value.numpy()


def _field_magnitude(result: dict[str, torch.Tensor], key: str) -> np.ndarray:
    value = _last(result, key)
    return torch.linalg.vector_norm(value, dim=0).numpy()


def save_panel(
    frames: torch.Tensor,
    result: dict[str, torch.Tensor],
    output: Path,
    *,
    title: str,
    ground_truth: torch.Tensor | None = None,
) -> None:
    """Save a headless 4x4 diagnostic panel."""

    fig, axes = plt.subplots(4, 4, figsize=(18, 15), constrained_layout=True)
    image_axes = axes.flat
    entries: list[tuple[str, Any, str]] = [
        ("RGB start", _image(frames, 0), "image"),
        ("RGB end", _image(frames, -1), "image"),
        ("ON motion energy", _map(result, "on_motion_energy"), "magma"),
        ("OFF motion energy", _map(result, "off_motion_energy"), "magma"),
        ("Observed energy", _map(result, "observed_total_energy"), "magma"),
        ("Observed pseudo-flow", _field_magnitude(result, "observed_pseudo_flow"), "viridis"),
        ("Predicted rotation flow", _field_magnitude(result, "predicted_rotation_flow"), "viridis"),
        ("Explained rotation energy", _map(result, "explained_total_energy"), "magma"),
        ("Residual energy", _map(result, "residual_total_energy"), "inferno"),
        ("Residual pseudo-flow", _field_magnitude(result, "residual_pseudo_flow"), "plasma"),
        ("Residual scale 1", _last(result, "residual_energy_by_scale")[0].numpy(), "inferno"),
        ("Residual scale 2", _last(result, "residual_energy_by_scale")[1].numpy(), "inferno"),
        ("Residual scale 3", _last(result, "residual_energy_by_scale")[2].numpy(), "inferno"),
    ]
    for axis, (label, value, cmap) in zip(image_axes, entries):
        axis.imshow(value, cmap=None if cmap == "image" else cmap)
        axis.set_title(label)
        axis.axis("off")

    prediction = _last(result, "predicted_rotation")
    text_axis = image_axes[13]
    text_axis.axis("off")
    text_axis.set_title("Rotation and ground truth")
    text = [f"predicted: {prediction.tolist()}"]
    if ground_truth is not None:
        text.append(f"ground truth: {ground_truth.detach().cpu().tolist()}")
    text.extend(
        [
            f"residual ratio: {_last(result, 'residual_ratio').item():.4f}",
            f"spatial support: {_last(result, 'spatial_support').item():.4f}",
            f"scale agreement: {_last(result, 'scale_agreement').item():.4f}",
            f"temporal agreement: {_last(result, 'temporal_agreement').item():.4f}",
        ]
    )
    text_axis.text(0.02, 0.98, "\n".join(text), va="top", family="monospace", fontsize=9)

    confidence_axis = image_axes[14]
    confidence_axis.axis("off")
    confidence_axis.set_title("Uncalibrated reliability")
    confidence_text = [
        f"old confidence: {_last(result, 'old_confidence').item():.4f}" if "old_confidence" in result else "old confidence: n/a",
        f"new global: {_last(result, 'global_confidence').item():.4f}",
        f"ON/OFF: {_last(result, 'on_off_agreement').item():.4f}",
        f"observability: {_last(result, 'axis_confidence').tolist()}",
        f"condition: {_last(result, 'observability_condition').item():.2f}",
        f"motion presence: {_last(result, 'motion_presence').item():.4f}",
    ]
    confidence_axis.text(0.02, 0.98, "\n".join(confidence_text), va="top", family="monospace", fontsize=9)

    energy_axis = image_axes[15]
    energy_axis.axis("off")
    energy_axis.set_title("Energy accounting")
    observed = _last(result, "observed_total_energy").sum().item()
    explained = _last(result, "explained_total_energy").sum().item()
    residual = _last(result, "residual_total_energy").sum().item()
    energy_axis.bar(["observed", "explained", "residual"], [observed, explained, residual], color=["#777", "#2878b5", "#d95f02"])
    energy_axis.tick_params(axis="x", labelrotation=25)
    energy_axis.set_ylabel("native energy units")

    fig.suptitle(title)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=130)
    plt.close(fig)


def _model_from_checkpoint(checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    sys.path.insert(0, str(Path(__file__).parent))
    from train import build_model

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = build_model(checkpoint.get("config", {}).get("model", "flyrot_motion_direct")).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def save_rgb_video(frames: torch.Tensor, output: Path, fps: int = 4) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames.shape[-2:]
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"could not open video writer: {output}")
    try:
        for index in range(len(frames)):
            frame = (_image(frames, index)[:, :, ::-1] * 255.0).astype(np.uint8)
            writer.write(frame)
    finally:
        writer.release()


def run_synthetic(args: argparse.Namespace, model: torch.nn.Module | None) -> dict[str, Any]:
    output_root = args.output / "synthetic"
    cases = ["zero", "rotation", "brightness", "translation", "moving_patch", "mixed"]
    metrics: dict[str, Any] = {}
    failures: list[str] = []
    for case_name in cases:
        case: SyntheticCase = make_synthetic_case(case_name, args.image_size, args.image_size, args.seed)
        oracle = build_rotation_residual_diagnostics(
            case.pair_rotations.unsqueeze(0),
            case.oracle_energy,
            torch.ones_like(case.oracle_energy),
            on_energy=case.oracle_on_energy,
            off_energy=case.oracle_off_energy,
        ).as_dict()
        case_metrics: dict[str, Any] = {
            "oracle": {
                "residual_ratio": _last(oracle, "residual_ratio"),
                "spatial_support": _last(oracle, "spatial_support"),
                "global_confidence": _last(oracle, "global_confidence"),
            }
        }
        save_panel(
            case.frames,
            oracle,
            output_root / f"{case_name}_oracle.png",
            title=f"Synthetic {case_name}: ground-truth rotation explanation",
            ground_truth=case.pair_rotations[-1],
        )
        wrong_rotation = -case.pair_rotations.unsqueeze(0)
        wrong = build_rotation_residual_diagnostics(
            wrong_rotation,
            case.oracle_energy,
            torch.ones_like(case.oracle_energy),
            on_energy=case.oracle_on_energy,
            off_energy=case.oracle_off_energy,
        )
        case_metrics["oracle"]["wrong_rotation_residual_ratio"] = _last(wrong.as_dict(), "residual_ratio")
        if model is not None:
            with torch.no_grad():
                output = model(case.frames.unsqueeze(0).to(args.device), diagnostics=True)
            model_result = {key: value.detach().cpu() for key, value in output.items() if isinstance(value, torch.Tensor)}
            save_panel(
                case.frames,
                model_result,
                output_root / f"{case_name}_model.png",
                title=f"Synthetic {case_name}: model diagnostic",
                ground_truth=case.pair_rotations[-1],
            )
            case_metrics["model"] = {
                key: _last(model_result, key)
                for key in (
                    "residual_ratio",
                    "spatial_support",
                    "scale_agreement",
                    "temporal_agreement",
                    "on_off_agreement",
                    "global_confidence",
                    "old_confidence",
                    "observability_condition",
                )
            }
        oracle_ratio = float(_last(oracle, "residual_ratio").mean().item())
        oracle_support = float(_last(oracle, "spatial_support").mean().item())
        if case_name == "rotation" and oracle_ratio >= 0.25:
            failures.append(f"rotation oracle residual too high: {oracle_ratio}")
        if case_name in {"zero", "brightness", "translation", "moving_patch"} and oracle_ratio < 0.95:
            failures.append(f"{case_name} oracle was incorrectly explained by rotation: {oracle_ratio}")
        if case_name == "moving_patch" and oracle_support >= 0.6:
            failures.append(f"moving patch support was not localized: {oracle_support}")
        metrics[case_name] = case_metrics

        if args.save_video:
            save_rgb_video(case.frames, output_root.parent / "videos" / f"{case_name}.mp4")
    metrics["failures"] = failures
    metrics_path = output_root.parent / "metrics" / "synthetic.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(_json_value(metrics), indent=2) + "\n", encoding="utf-8")
    if failures:
        raise RuntimeError("synthetic acceptance failures: " + "; ".join(failures))
    return metrics


def _real_item(
    args: argparse.Namespace,
    model: torch.nn.Module,
    dataset: TartanAirWindowDataset,
    record_index: int,
    sample_index: int,
) -> dict[str, Any]:
    sample = dataset[sample_index]
    with torch.no_grad():
        output = model(sample["frames"].unsqueeze(0).to(args.device), diagnostics=True)
    result = {key: value.detach().cpu() for key, value in output.items() if isinstance(value, torch.Tensor)}
    prediction = result["rotation_vector"][:, -1]
    target = sample["target_rotation_vector"].view(1, 3)
    metrics = rotation_metrics(prediction, target, result["old_confidence"][:, -1])
    metrics.update(
        {
            "environment": sample["metadata"]["environment"],
            "trajectory": sample["metadata"]["trajectory"],
            "record_index": record_index,
            "sample_index": sample_index,
            "residual_ratio": _last(result, "residual_ratio"),
            "spatial_support": _last(result, "spatial_support"),
            "scale_agreement": _last(result, "scale_agreement"),
            "temporal_agreement": _last(result, "temporal_agreement"),
            "on_off_agreement": _last(result, "on_off_agreement"),
            "axis_confidence": _last(result, "axis_confidence"),
            "global_confidence": _last(result, "global_confidence"),
            "observability_condition": _last(result, "observability_condition"),
            "frames": sample["frames"],
            "result": result,
            "target": sample["target_rotation_vector"],
        }
    )
    return metrics


def run_real(args: argparse.Namespace, model: torch.nn.Module) -> dict[str, Any]:
    if args.index is None:
        raise ValueError("--index is required for real diagnostics")
    records = default_split(json.loads(args.index.read_text(encoding="utf-8")))[args.split]
    if not records:
        raise ValueError(f"no records in split {args.split}")
    if not 0 <= args.record_index < len(records):
        raise IndexError(f"record index {args.record_index} outside split of length {len(records)}")

    datasets = {
        index: TartanAirWindowDataset(
            [record],
            window_length=args.window_length,
            frame_gap=args.frame_gap,
            image_size=args.image_size,
            target_direction=args.target_direction,
            preload_images=args.preload_images,
            preload_workers=args.preload_workers,
        )
        for index, record in enumerate(records)
    }
    if args.sample_count <= 1:
        if not 0 <= args.sample_index < len(datasets[args.record_index]):
            raise IndexError(f"sample index {args.sample_index} outside dataset")
        selections = [(args.record_index, args.sample_index)]
    else:
        per_record = max(1, (args.sample_count + len(records) - 1) // len(records))
        selections = []
        for record_index, dataset in datasets.items():
            if len(dataset) == 1:
                indices = [0]
            else:
                indices = sorted(
                    {
                        round(index * (len(dataset) - 1) / max(1, per_record - 1))
                        for index in range(per_record)
                    }
                )
            selections.extend((record_index, index) for index in indices)
        selections = selections[: args.sample_count]

    items = [_real_item(args, model, datasets[record_index], record_index, sample_index) for record_index, sample_index in selections]
    items.sort(key=lambda item: float(item["mean_geodesic_deg"]))
    chosen = {
        "best": items[0],
        "median": items[len(items) // 2],
        "worst": items[-1],
    }
    for label, item in chosen.items():
        save_panel(
            item["frames"],
            item["result"],
            args.output / "real" / f"{args.split}_{label}_record{item['record_index']}_sample{item['sample_index']}.png",
            title=f"Real {args.split} {label}: {item['environment']}/{item['trajectory']}",
            ground_truth=item["target"],
        )

    predictions = torch.cat([item["result"]["rotation_vector"][:, -1] for item in items])
    targets = torch.stack([item["target"] for item in items])
    confidence = torch.cat([item["result"]["old_confidence"][:, -1, 0] for item in items])
    aggregate = rotation_metrics(predictions, targets, confidence)
    aggregate.update(
        {
            "split": args.split,
            "samples": len(items),
            "diagnostic_mean_residual_ratio": float(torch.stack([_last(item["result"], "residual_ratio") for item in items]).mean().item()),
            "diagnostic_mean_spatial_support": float(torch.stack([_last(item["result"], "spatial_support") for item in items]).mean().item()),
            "diagnostic_mean_scale_agreement": float(torch.stack([_last(item["result"], "scale_agreement") for item in items]).mean().item()),
            "diagnostic_mean_temporal_agreement": float(torch.stack([_last(item["result"], "temporal_agreement") for item in items]).mean().item()),
            "diagnostic_mean_on_off_agreement": float(torch.stack([_last(item["result"], "on_off_agreement") for item in items]).mean().item()),
            "diagnostic_mean_global_confidence": float(torch.stack([_last(item["result"], "global_confidence") for item in items]).mean().item()),
        }
    )
    report = {
        "aggregate": aggregate,
        "best": {key: value for key, value in chosen["best"].items() if key not in {"frames", "result", "target"}},
        "median": {key: value for key, value in chosen["median"].items() if key not in {"frames", "result", "target"}},
        "worst": {key: value for key, value in chosen["worst"].items() if key not in {"frames", "result", "target"}},
    }
    output_path = args.output / "metrics" / f"{args.split}_summary.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(_json_value(report), indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("models/flyrot_v0_best.pt"))
    parser.add_argument("--index", type=Path)
    parser.add_argument("--output", type=Path, default=Path("artifacts/rotation_residual"))
    parser.add_argument("--device", default=None)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--window-length", type=int, default=3)
    parser.add_argument("--frame-gap", type=int, default=4)
    parser.add_argument("--target-direction", default="optical_image_motion")
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--record-index", type=int, default=0)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--sample-count", type=int, default=3)
    parser.add_argument("--preload-images", action="store_true")
    parser.add_argument("--preload-workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--synthetic-smoke-test", action="store_true")
    parser.add_argument("--save-video", action="store_true")
    args = parser.parse_args()
    args.device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    model = _model_from_checkpoint(args.checkpoint, args.device) if args.checkpoint.is_file() else None
    if args.synthetic_smoke_test:
        run_synthetic(args, model)
    elif model is None:
        raise FileNotFoundError(args.checkpoint)
    else:
        result = run_real(args, model)
        print(json.dumps(_json_value(result), indent=2))


if __name__ == "__main__":
    main()
