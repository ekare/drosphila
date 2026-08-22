#!/usr/bin/env python3
"""Trajectory-balanced directional-residual and oracle evaluator."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
from scipy.stats import spearmanr
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).parent))
from train import build_model  # noqa: E402

from flyrot.data.tartanair import TartanAirWindowDataset, default_split  # noqa: E402
from flyrot.diagnostics import build_rotation_residual_diagnostics  # noqa: E402
from flyrot.metrics import per_sample_geodesic_rad  # noqa: E402
from flyrot.geometry.rotational_flow import CameraIntrinsics  # noqa: E402


def _json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return _json_value(value.detach().cpu().tolist())
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, (np.floating, np.integer)):
        return _json_value(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def balanced_indices(lengths: list[int], total: int) -> list[list[int]]:
    """Select deterministic evenly spaced windows with balanced records."""

    if total < 1:
        raise ValueError("total must be positive")
    total_available = sum(lengths)
    requested = min(total, total_available)
    counts = [min(length, requested // len(lengths) + int(index < requested % len(lengths))) for index, length in enumerate(lengths)]
    while sum(counts) < requested:
        for index, length in enumerate(lengths):
            if counts[index] < length:
                counts[index] += 1
                if sum(counts) == requested:
                    break
    selected: list[list[int]] = []
    for length, count in zip(lengths, counts):
        if count >= length:
            selected.append(list(range(length)))
        elif count == 1:
            selected.append([0])
        else:
            selected.append(sorted({round(step * (length - 1) / (count - 1)) for step in range(count)}))
    return selected


def _scalar(value: torch.Tensor) -> torch.Tensor:
    return value.reshape(value.shape[0], -1)[:, -1]


def _finite_values(values: list[float]) -> torch.Tensor:
    tensor = torch.tensor(values, dtype=torch.float64)
    return tensor[torch.isfinite(tensor)]


def _distribution(values: list[float]) -> dict[str, Any]:
    finite = _finite_values(values)
    if not len(finite):
        return {key: None for key in ("mean", "median", "p25", "p75", "p90", "histogram", "cdf")}
    minimum = float(finite.min())
    maximum = float(finite.max())
    if minimum < 0.0:
        padding = max(1e-6, (maximum - minimum) * 0.025)
        bins = np.linspace(minimum - padding, maximum + padding, 21)
    else:
        bins = np.linspace(0.0, max(1.0, maximum * 1.05), 21)
    counts, edges = np.histogram(finite.numpy(), bins=bins)
    sorted_values = torch.sort(finite).values
    cdf = []
    for percentile in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0):
        index = min(len(sorted_values) - 1, round(percentile * (len(sorted_values) - 1)))
        cdf.append({"quantile": percentile, "value": float(sorted_values[index])})
    return {
        "mean": float(finite.mean()),
        "median": float(finite.median()),
        "p25": float(torch.quantile(finite, 0.25)),
        "p75": float(torch.quantile(finite, 0.75)),
        "p90": float(torch.quantile(finite, 0.90)),
        "histogram": {"edges": edges.tolist(), "counts": counts.tolist()},
        "cdf": cdf,
    }


def _spearman(error: list[float], signal: list[float]) -> float | None:
    x = np.asarray(error, dtype=np.float64)
    y = np.asarray(signal, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3 or np.std(x[mask]) == 0 or np.std(y[mask]) == 0:
        return None
    return float(spearmanr(x[mask], y[mask]).statistic)


def _risk_coverage(error_deg: list[float], score: list[float]) -> tuple[list[dict[str, Any]], float | None]:
    error = np.asarray(error_deg, dtype=np.float64)
    signal = np.asarray(score, dtype=np.float64)
    mask = np.isfinite(error) & np.isfinite(signal)
    if mask.sum() == 0:
        return [], None
    ranked = error[mask][np.argsort(signal[mask])[::-1]]
    cumulative = np.cumsum(ranked) / np.arange(1, len(ranked) + 1)
    aurc = float(cumulative.mean())
    result = []
    for coverage in (0.10, 0.25, 0.50, 0.75, 1.0):
        count = max(1, min(len(ranked), math.ceil(len(ranked) * coverage)))
        result.append({"coverage": coverage, "samples": count, "risk_geodesic_deg": float(ranked[:count].mean())})
    return result, aurc


def _auroc(error_deg: list[float], score: list[float], threshold: float) -> float | None:
    error = np.asarray(error_deg, dtype=np.float64)
    signal = np.asarray(score, dtype=np.float64)
    mask = np.isfinite(error) & np.isfinite(signal)
    labels = error[mask] <= threshold
    if labels.sum() == 0 or labels.sum() == len(labels):
        return None
    order = np.argsort(signal[mask], kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(order) + 1)
    positive = labels
    return float((ranks[positive].sum() - positive.sum() * (positive.sum() + 1) / 2) / (positive.sum() * (~positive).sum()))


def _group_summary(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row[key]), []).append(row)
    result = {}
    for group, members in sorted(groups.items()):
        errors = [item["geodesic_error_deg"] for item in members]
        result[group] = {
            "samples": len(members),
            "mean_geodesic_deg": float(np.mean(errors)),
            "mean_directional_residual_pred": float(np.mean([item["directional_residual_pred"] for item in members])),
            "mean_directional_residual_gt": float(np.mean([item["directional_residual_gt"] for item in members])),
        }
    return result


def _macro_bootstrap(rows: list[dict[str, Any]], key: str, seed: int = 0, replicates: int = 1000) -> dict[str, Any]:
    """Bootstrap the unweighted mean of per-group geodesic means."""

    groups: dict[str, list[float]] = {}
    for row in rows:
        groups.setdefault(str(row[key]), []).append(float(row["geodesic_error_deg"]))
    group_means = np.asarray([np.mean(values) for _, values in sorted(groups.items())], dtype=np.float64)
    if len(group_means) == 0:
        return {"groups": 0, "mean_geodesic_deg": None, "bootstrap_95ci_deg": None}
    generator = np.random.default_rng(seed)
    draws = generator.choice(group_means, size=(replicates, len(group_means)), replace=True).mean(axis=1)
    return {
        "groups": int(len(group_means)),
        "mean_geodesic_deg": float(group_means.mean()),
        "bootstrap_95ci_deg": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
        "bootstrap_seed": seed,
        "bootstrap_replicates": replicates,
    }


def _validity_reason_codes(row: dict[str, Any]) -> list[str]:
    reasons = []
    if not row["scale_agreement_valid"]:
        reasons.append("scale_agreement_unavailable")
    if not row["temporal_agreement_valid"]:
        reasons.append("temporal_agreement_unavailable_for_current_state")
    if not row["on_off_agreement_valid"]:
        reasons.append("on_off_agreement_unavailable")
    if not row["observability_valid"]:
        reasons.append("observability_invalid")
    if not row["global_reliability_valid"]:
        reasons.append("global_reliability_invalid")
    if abs(row["directional_residual_wrong_sign"] - row["directional_residual_gt"]) < 0.10:
        reasons.append("wrong_sign_oracle_not_discriminative")
    return reasons


def _summary(rows: list[dict[str, Any]], split: str, target_count: int, intrinsics_mode: str) -> dict[str, Any]:
    errors = [row["geodesic_error_deg"] for row in rows]
    fields = {
        "directional_residual_pred": [row["directional_residual_pred"] for row in rows],
        "directional_residual_gt": [row["directional_residual_gt"] for row in rows],
        "directional_residual_zero": [row["directional_residual_zero"] for row in rows],
        "directional_residual_wrong_sign": [row["directional_residual_wrong_sign"] for row in rows],
        "directional_residual_delta": [row["directional_residual_delta"] for row in rows],
        "old_confidence": [row["old_confidence"] for row in rows],
        "global_reliability": [row["global_reliability"] for row in rows],
        "directional_fit_score": [row["directional_fit_score"] for row in rows],
    }
    availability_fields = [
        "scale_agreement_valid",
        "temporal_agreement_valid",
        "on_off_agreement_valid",
        "observability_valid",
        "global_reliability_valid",
    ]
    risk_signals = {
        "old_confidence": fields["old_confidence"],
        "global_reliability": fields["global_reliability"],
        "directional_fit_score": fields["directional_fit_score"],
        "inverse_directional_residual_pred": [-value for value in fields["directional_residual_pred"]],
        "inverse_directional_residual_delta_oracle": [-abs(value) for value in fields["directional_residual_delta"]],
    }
    risks = {}
    for name, signal in risk_signals.items():
        curve, aurc = _risk_coverage(errors, signal)
        risks[name] = {"aurc_geodesic_deg": aurc, "risk_coverage": curve}
    risks["auroc_good_prediction"] = {
        name: {f"error_le_{threshold:g}_deg": _auroc(errors, signal, threshold) for threshold in (5.0, 10.0, 15.0)}
        for name, signal in risk_signals.items()
    }
    return {
        "split": split,
        "samples": len(rows),
        "requested_samples": target_count,
        "trajectory_balanced": True,
        "intrinsics_mode": intrinsics_mode,
        "geodesic_error_deg": _distribution(errors),
        "directional_residual_distribution": {key: _distribution(value) for key, value in fields.items()},
        "spearman_error_vs": {
            key: _spearman(errors, value)
            for key, value in {
                "directional_residual_pred": fields["directional_residual_pred"],
                "directional_residual_delta": fields["directional_residual_delta"],
                "old_confidence": fields["old_confidence"],
                "global_reliability": fields["global_reliability"],
            }.items()
        },
        "risk_and_coverage": risks,
        "availability": {
            key: float(np.mean([bool(row[key]) for row in rows])) for key in availability_fields
        },
        "validity_reason_counts": {
            reason: sum(reason in row["validity_reason_codes"] for row in rows)
            for reason in sorted({reason for row in rows for reason in row["validity_reason_codes"]})
        },
        "oracle_contract": {
            "wrong_sign_separation_mean": float(
                np.mean([row["directional_residual_wrong_sign"] - row["directional_residual_gt"] for row in rows])
            )
            if rows
            else None,
            "wrong_sign_separation_median": float(
                np.median([row["directional_residual_wrong_sign"] - row["directional_residual_gt"] for row in rows])
            )
            if rows
            else None,
            "required_median_separation": 0.10,
            "status": "pass"
            if rows
            and float(np.median([row["directional_residual_wrong_sign"] - row["directional_residual_gt"] for row in rows])) >= 0.10
            else "unresolved",
        },
        "macro_bootstrap": {
            "trajectory": _macro_bootstrap(rows, "trajectory"),
            "environment": _macro_bootstrap(rows, "environment"),
        },
        "groups": {
            "environment": _group_summary(rows, "environment"),
            "trajectory": _group_summary(rows, "trajectory"),
            "rotation_magnitude_bin": _group_summary(rows, "rotation_magnitude_bin"),
            "motion_energy_bin": _group_summary(rows, "motion_energy_bin"),
            "spatial_support_bin": _group_summary(rows, "spatial_support_bin"),
            "axis_bin": _group_summary(rows, "axis_bin"),
        },
    }


def _bin(value: float, edges: tuple[float, ...], labels: tuple[str, ...]) -> str:
    for edge, label in zip(edges, labels):
        if value < edge:
            return label
    return labels[-1]


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = build_model(checkpoint.get("config", {}).get("model", "flyrot_motion_direct")).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    records = default_split(json.loads(args.index.read_text(encoding="utf-8")))[args.split]
    datasets = [
        TartanAirWindowDataset(
            [record],
            window_length=args.window_length,
            frame_gap=args.frame_gap,
            image_size=args.image_size,
            target_direction=args.target_direction,
            preload_images=args.preload_images,
            preload_workers=args.preload_workers,
        )
        for record in records
    ]
    selections = balanced_indices([len(dataset) for dataset in datasets], args.samples)
    intrinsics = None
    intrinsics_mode = "legacy_centered"
    if args.intrinsics:
        intrinsics = CameraIntrinsics.from_mapping(json.loads(args.intrinsics.read_text(encoding="utf-8")))
        intrinsics_mode = "full_K"
    rows: list[dict[str, Any]] = []
    for record_index, (record, dataset, indices) in enumerate(zip(records, datasets, selections)):
        loader = DataLoader(Subset(dataset, indices), batch_size=args.batch_size, shuffle=False, num_workers=0)
        for batch_offset, batch in enumerate(loader):
            frames = batch["frames"].to(device)
            targets = batch["target_rotation_vector"].to(device)
            with torch.no_grad():
                output = model(frames, diagnostics=True)
            energy = output["direction_energy"][:, -1:]
            valid = output["valid_mask"][:, -1:]
            on_energy = output.get("on_motion_energy", output.get("on_energy"))[:, -1:]
            off_energy = output.get("off_motion_energy", output.get("off_energy"))[:, -1:]
            predictions = output["rotation_vector"][:, -1:]
            gt_diag = build_rotation_residual_diagnostics(
                targets.unsqueeze(1), energy, valid, on_energy=on_energy, off_energy=off_energy, intrinsics=intrinsics,
            ).as_dict()
            zero_diag = build_rotation_residual_diagnostics(
                torch.zeros_like(targets).unsqueeze(1), energy, valid, on_energy=on_energy, off_energy=off_energy, intrinsics=intrinsics,
            ).as_dict()
            wrong_diag = build_rotation_residual_diagnostics(
                (-targets).unsqueeze(1), energy, valid, on_energy=on_energy, off_energy=off_energy, intrinsics=intrinsics,
            ).as_dict()
            pred_error = torch.rad2deg(per_sample_geodesic_rad(predictions[:, 0], targets))
            for item_index in range(len(targets)):
                metadata = {key: value[item_index] for key, value in batch["metadata"].items()}
                residual_pred = float(output["directional_residual_ratio"][item_index, -1, 0].detach().cpu())
                residual_gt = float(gt_diag["directional_residual_ratio"][item_index, 0, 0].detach().cpu())
                residual_zero = float(zero_diag["directional_residual_ratio"][item_index, 0, 0].detach().cpu())
                residual_wrong = float(wrong_diag["directional_residual_ratio"][item_index, 0, 0].detach().cpu())
                support = float(output["spatial_support_score"][item_index, -1, 0].detach().cpu())
                energy_total = float(output["observed_total_energy"][item_index, -1].sum().detach().cpu())
                target_vector = targets[item_index]
                prediction_vector = predictions[item_index, 0]
                target_norm = float(torch.linalg.vector_norm(target_vector).detach().cpu())
                prediction_norm = float(torch.linalg.vector_norm(prediction_vector).detach().cpu())
                if target_norm < 0.02:
                    axis_bin = "near_zero"
                    axis_cosine = None
                else:
                    dominant = int(torch.argmax(target_vector.abs()).detach().cpu())
                    axis_name = ("x", "y", "z")[dominant]
                    axis_bin = f"{'+' if float(target_vector[dominant]) >= 0 else '-'}{axis_name}"
                    axis_cosine = (
                        float(torch.dot(target_vector, prediction_vector).detach().cpu()) / (target_norm * prediction_norm)
                        if prediction_norm > 1e-8
                        else None
                    )
                row = {
                        "record_index": record_index,
                        "sample_index": int(indices[batch_offset * args.batch_size + item_index]),
                        "environment": metadata["environment"],
                        "trajectory": metadata["trajectory"],
                        "geodesic_error_deg": float(pred_error[item_index].cpu()),
                        "rotation_magnitude_rad": float(torch.linalg.vector_norm(targets[item_index]).cpu()),
                        "rotation_magnitude_bin": _bin(float(torch.linalg.vector_norm(targets[item_index]).cpu()), (0.05, 0.15, 0.30), ("small", "medium", "large")),
                        "motion_energy": energy_total,
                        "motion_energy_bin": _bin(energy_total, (1.0, 10.0, 100.0), ("low", "medium", "high")),
                        "spatial_support": support,
                        "spatial_support_bin": _bin(support, (0.5, 0.8, 0.95), ("localized", "mixed", "broad")),
                        "axis_bin": axis_bin,
                        "axis_cosine": axis_cosine,
                        "directional_residual_pred": residual_pred,
                        "directional_residual_gt": residual_gt,
                        "directional_residual_zero": residual_zero,
                        "directional_residual_wrong_sign": residual_wrong,
                        "directional_residual_delta": residual_pred - residual_gt,
                        "directional_fit_score": float(output["directional_fit_score"][item_index, -1, 0].detach().cpu()),
                        "old_confidence": float(output["old_confidence"][item_index, -1, 0].detach().cpu()),
                        "global_reliability": float(output["global_reliability"][item_index, -1, 0].detach().cpu()),
                        "scale_agreement_valid": bool(output["scale_agreement_valid"][item_index, -1, 0].detach().cpu()),
                        "temporal_agreement_valid": bool(output["temporal_agreement_valid"][item_index, -1, 0].detach().cpu()),
                        "on_off_agreement_valid": bool(output["on_off_agreement_valid"][item_index, -1, 0].detach().cpu()),
                        "observability_valid": bool(output["observability_valid"][item_index, -1, 0].detach().cpu()),
                        "global_reliability_valid": bool(output["global_reliability_valid"][item_index, -1, 0].detach().cpu()),
                        "temporal_agreement_score": _json_value(output["temporal_agreement_score"][item_index, -1, 0]),
                        "on_off_agreement_score": _json_value(output["on_off_agreement_score"][item_index, -1, 0]),
                        "scale_agreement_score": _json_value(output["scale_agreement_score"][item_index, -1, 0]),
                    }
                row["validity_reason_codes"] = _validity_reason_codes(row)
                rows.append(row)
    result = {
        "checkpoint_config": checkpoint.get("config", {}),
        "split": args.split,
        "target_direction": args.target_direction,
        "window_length": args.window_length,
        "frame_gap": args.frame_gap,
        "rows": rows,
        "summary": _summary(rows, args.split, args.samples, intrinsics_mode),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(_json_value(result), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--window-length", type=int, default=3)
    parser.add_argument("--frame-gap", type=int, default=4)
    parser.add_argument("--target-direction", default="optical_image_motion")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--preload-images", action="store_true")
    parser.add_argument("--preload-workers", type=int, default=1)
    parser.add_argument("--intrinsics", type=Path)
    args = parser.parse_args()
    result = evaluate(args)
    print(json.dumps(_json_value(result["summary"]), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
