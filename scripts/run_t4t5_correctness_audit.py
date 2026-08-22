#!/usr/bin/env python3
"""Run the bounded T4/T5 correctness audit and corrected field evaluation.

This script is evaluator-only. It does not train, alter checkpoint weights, or
write into the source dataset. Runtime output is intended for the operator's
NV1 evidence directory, not for a public repository commit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import spearmanr

from flyrot.data.tartanair import default_split
from flyrot.geometry.camera import tartanair_v2_lcam_front_intrinsics
from flyrot.models.direction_cells import DirectionCellBank
from flyrot.models.flyrot_v1 import FlyRotV1
from flyrot.real_texture import (
    RealTextureRotationDataset,
    decode_direction_population,
    manifest_sha256,
    path_free_record,
    solve_weighted_rotation,
)
from flyrot.t4t5 import ANGLES_DEG, CONTRASTS, DISPLACEMENTS, STIMULUS_CLASSES, StimulusSpec, make_stimulus
from flyrot.validation_contract import (
    CANONICAL_DIRECTION_NAMES,
    d4_audit,
    direction_index,
    photoreceptor_variants,
    pure_edge_stimulus,
    reorder_direction_vector,
    response_metrics,
)


def _commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _sha256(path: Path | None) -> str | None:
    if path is None or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(value: object) -> object:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json(item) for item in value]
    return value


def _is_motion(kind: str) -> bool:
    return kind.startswith("moving_") or kind in {"contrast_reversed_edge", "aperture_limited_edge"}


def _bank_pooled(on: torch.Tensor, off: torch.Tensor, valid_spatial: torch.Tensor, bank: DirectionCellBank) -> dict[str, torch.Tensor]:
    energy, valid, on_energy, off_energy = bank(on, off, return_components=True)
    b, steps, _, height, width = energy.shape
    scales = tuple(bank.scales)
    on_by = on_energy.reshape(b, steps, len(scales), 8, height, width)
    off_by = off_energy.reshape(b, steps, len(scales), 8, height, width)
    valid_by = valid.reshape(b, steps, len(scales), 8, height, width)
    spatial = valid_spatial.reshape(1, 1, 1, 1, height, width)

    def pooled(values: torch.Tensor) -> torch.Tensor:
        weights = valid_by * spatial
        return (values * weights).sum(dim=(-1, -2)) / weights.sum(dim=(-1, -2)).clamp_min(1.0)

    on_by_scale = pooled(on_by).mean(dim=1)[0]
    off_by_scale = pooled(off_by).mean(dim=1)[0]
    on_legacy = on_by_scale.mean(dim=0)
    off_legacy = off_by_scale.mean(dim=0)
    response_legacy = on_legacy + off_legacy
    return {
        "on_response_legacy": on_legacy,
        "off_response_legacy": off_legacy,
        "response_legacy": response_legacy,
        "on_response": reorder_direction_vector(on_legacy),
        "off_response": reorder_direction_vector(off_legacy),
        "response": reorder_direction_vector(response_legacy),
        "on_response_by_scale": reorder_direction_vector(on_by_scale),
        "off_response_by_scale": reorder_direction_vector(off_by_scale),
        "valid_fraction": valid_by.mul(spatial).mean(),
    }


def _t4_case(spec: StimulusSpec, scales: tuple[int, ...], device: torch.device, bank: DirectionCellBank) -> dict[str, object]:
    frames, valid = make_stimulus(spec)
    with torch.inference_mode():
        variants = photoreceptor_variants(frames.to(device))
        p0 = variants["P0"]
        result = _bank_pooled(p0["on"], p0["off"], valid.to(device), bank)
    expected = direction_index(spec.angle_deg)
    metrics = response_metrics(result["response"].unsqueeze(0), torch.tensor([expected], device=device))
    return _json(
        {
            "kind": spec.kind,
            "angle_deg": spec.angle_deg,
            "direction_index": expected,
            "direction_name": CANONICAL_DIRECTION_NAMES[expected],
            "displacement": spec.displacement,
            "contrast": spec.contrast_name,
            "gamma": spec.gamma,
            "brightness_offset": spec.brightness_offset,
            "noise_std": spec.noise_std,
            "blur_sigma": spec.blur_sigma,
            "metrics": metrics,
            "preferred_index": int(result["response"].argmax()),
            **result,
        }
    )


def _grid_specs(*, width: int, height: int, small: bool) -> list[StimulusSpec]:
    rows: list[StimulusSpec] = []
    displacements = (1.0, 4.0, 16.0) if small else DISPLACEMENTS
    for kind in STIMULUS_CLASSES:
        motion = _is_motion(kind)
        angles = ANGLES_DEG if motion else (0,)
        values = displacements if motion else (0.0,)
        contrasts = tuple(CONTRASTS) if kind not in {"stationary_uniform", "brightness_increase", "brightness_decrease", "flicker"} else ("medium",)
        for angle in angles:
            for displacement in values:
                for contrast in contrasts:
                    rows.append(StimulusSpec(kind=kind, angle_deg=angle, displacement=displacement, contrast_name=contrast, height=height, width=width, seed=20260823))
    return rows


def _run_t4_grid(output: Path, scales: tuple[int, ...], device: torch.device, *, small: bool) -> dict[str, object]:
    height, width = (256, 288)
    bank = DirectionCellBank(scales=scales).to(device).eval()
    rows = [_t4_case(spec, scales, device, bank) for spec in _grid_specs(width=width, height=height, small=small)]
    manifest = {"kind": "corrected-t4t5-grid", "small": small, "raster": [width, height], "steps": 5, "scales": list(scales), "angles_deg": list(ANGLES_DEG), "displacements": list(DISPLACEMENTS if not small else (1.0, 4.0, 16.0)), "canonical_order": list(CANONICAL_DIRECTION_NAMES), "test_split_selection": "not applicable"}
    report = {"experiment": manifest, "dataset_manifest_sha256": manifest_sha256(manifest), "commit": _commit(), "cases": rows}
    name = "t4t5_corrected_small_cases.json" if small else "t4t5_corrected_full_cases.json"
    (output / name).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _pure_polarity_audit(output: Path, device: torch.device, scales: tuple[int, ...]) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    bank = DirectionCellBank(scales=scales).to(device).eval()
    rows: list[dict[str, object]] = []
    traces: dict[str, object] = {}
    for kind in ("on", "off"):
        for angle in ANGLES_DEG:
            frames, valid = pure_edge_stimulus(kind, angle_deg=angle, displacement=4.0)
            with torch.inference_mode():
                variants = photoreceptor_variants(frames.to(device))
                for variant_name, variant in variants.items():
                    result = _bank_pooled(variant["on"], variant["off"], valid.to(device), bank)
                    expected = direction_index(angle)
                    metrics = response_metrics(result["response"].unsqueeze(0), torch.tensor([expected], device=device))
                    matched = float(result["on_response"].max() if kind == "on" else result["off_response"].max())
                    other = float(result["off_response"].max() if kind == "on" else result["on_response"].max())
                    rows.append(
                        _json(
                            {
                                "edge": kind,
                                "angle_deg": angle,
                                "direction_index": expected,
                                "variant": variant_name,
                                "metrics": metrics,
                                "matched_peak": matched,
                                "cross_polarity_peak": other,
                                "cross_polarity_leakage": other / max(matched, 1e-8),
                                "on_response": result["on_response"],
                                "off_response": result["off_response"],
                            }
                        )
                    )
                    if angle in {0, 90}:
                        pre = variant["pre"][0]
                        adapted = variant["adapted"][0]
                        center = (pre.shape[-2] // 2, pre.shape[-1] // 2)
                        traces[f"{kind}_{angle}_{variant_name}"] = {
                            "pre_center_trace": pre[:, 0, center[0], center[1]],
                            "adapted_center_trace": adapted[:, 0, center[0], center[1]],
                            "on_center_trace": variant["on"][0, :, 0, center[0], center[1]],
                            "off_center_trace": variant["off"][0, :, 0, center[0], center[1]],
                            "pre_mean_abs": pre.abs().mean(dim=(1, 2, 3)),
                            "adapted_mean_abs": adapted.abs().mean(dim=(1, 2, 3)),
                        }
    summary: dict[str, object] = {"variants": {}, "rows": rows, "traces": _json(traces), "pure_edge_only": True, "mixed_polarity_stimuli_excluded": ["moving_bright_bar", "moving_dark_bar", "moving_sinusoidal_grating", "contrast_reversed_edge", "moving_patch", "aperture_limited_edge"]}
    for variant in ("P0", "P1", "P2"):
        selected = [row for row in rows if row["variant"] == variant]
        on_rows = [row for row in selected if row["edge"] == "on"]
        off_rows = [row for row in selected if row["edge"] == "off"]
        summary["variants"][variant] = {
            "on_edge_off_leakage": float(np.median([row["cross_polarity_leakage"] for row in on_rows])),
            "off_edge_on_leakage": float(np.median([row["cross_polarity_leakage"] for row in off_rows])),
            "on_direction_median_error_deg": float(np.median([row["metrics"]["population_vector_median_error_deg"] for row in on_rows])),
            "off_direction_median_error_deg": float(np.median([row["metrics"]["population_vector_median_error_deg"] for row in off_rows])),
            "on_direction_cosine_sign_accuracy": float(np.mean([row["metrics"]["population_vector_cosine_sign_accuracy"] for row in on_rows])),
            "off_direction_cosine_sign_accuracy": float(np.mean([row["metrics"]["population_vector_cosine_sign_accuracy"] for row in off_rows])),
        }
    (output / "polarity_frontend_audit.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _scale_audit(output: Path, device: torch.device) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    requested = (0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 16.0)
    bank_scales = (1, 2, 4, 8, 12, 16)
    bank = DirectionCellBank(scales=bank_scales).to(device).eval()
    rows = []
    for displacement in requested:
        frames, valid = pure_edge_stimulus("on", angle_deg=0, displacement=displacement, height=256, width=288)
        with torch.inference_mode():
            variant = photoreceptor_variants(frames.to(device))["P0"]
            result = _bank_pooled(variant["on"], variant["off"], valid.to(device), bank)
        per_scale = result["on_response_by_scale"][:, 0].detach().cpu().numpy()
        total = float(result["on_response"][0])
        rows.append({"stimulus_displacement": displacement, "response_peak": float(per_scale.max()), "response_sum": total, "scale_response": {str(scale): float(value) for scale, value in zip(bank_scales, per_scale)}})
    peaks = np.asarray([row["response_peak"] for row in rows], dtype=float)
    rho = spearmanr(np.asarray(requested), peaks).statistic if len(peaks) > 1 else float("nan")
    peak_max = float(peaks.max()) if peaks.size else 0.0
    width = int(np.sum(peaks >= peak_max * 0.5)) if peak_max else 0
    result = {
        "requested_stimulus_scales": list(requested),
        "cell_bank_scales": list(bank_scales),
        "rows": rows,
        "response_peak_width_at_half_max": width,
        "dead_response": bool(peak_max <= 1e-8),
        "alias_ratio_second_to_peak": float(np.partition(peaks, -2)[-2] / max(peak_max, 1e-8)) if len(peaks) > 1 else None,
        "monotonicity_spearman": None if not np.isfinite(rho) else float(rho),
        "physical_pixel_calibration": "unavailable_un_calibrated",
        "no_px_claim": True,
        "calibration_split": "deterministic_pure_edge_only; no test split used",
    }
    (output / "magnitude_calibration_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def _mean_pool(values: torch.Tensor, weights: torch.Tensor, size: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor]:
    if weights.ndim == values.ndim - 1:
        weights = weights.unsqueeze(2)
    prefix = values.shape[:-2]
    channels = values.shape[-3]
    height, width = values.shape[-2:]
    value_4d = values.reshape(-1, channels, height, width)
    weight_4d = weights.expand(*prefix[:-1], 1, height, width).reshape(-1, 1, height, width)
    pooled_weight = torch.nn.functional.adaptive_avg_pool2d(weight_4d, size)
    pooled = torch.nn.functional.adaptive_avg_pool2d(value_4d * weight_4d, size) / pooled_weight.clamp_min(1e-8)
    return pooled.reshape(*prefix, size[0], size[1]), pooled_weight.reshape(*prefix[:-1], 1, size[0], size[1])


def _field_metrics(prediction: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> dict[str, float | int | None]:
    pred_flat = prediction.permute(0, 1, 3, 4, 2).reshape(-1, 2)
    target_flat = target.permute(0, 1, 3, 4, 2).reshape(-1, 2)
    valid_flat = valid.squeeze(2).reshape(-1) > 0.2
    pred_mag = torch.linalg.vector_norm(pred_flat, dim=-1)
    target_mag = torch.linalg.vector_norm(target_flat, dim=-1)
    observable = valid_flat & (target_mag > 0.5) & (pred_mag > 1e-5)
    if not bool(observable.any()):
        return {"observable_cells": 0, "median_angular_error_deg": None, "direction_sign_accuracy": None, "magnitude_spearman": None, "epe": float(torch.linalg.vector_norm(target_flat, dim=-1).mean()), "zero_epe": float(target_mag.mean()), "epe_improvement_vs_zero": 0.0}
    pred_obs, target_obs = pred_flat[observable], target_flat[observable]
    cosine = (pred_obs * target_obs).sum(-1) / (torch.linalg.vector_norm(pred_obs, dim=-1) * torch.linalg.vector_norm(target_obs, dim=-1)).clamp_min(1e-8)
    angular = torch.rad2deg(torch.acos(cosine.clamp(-1, 1)))
    rho = spearmanr(torch.linalg.vector_norm(pred_obs, dim=-1).cpu().numpy(), torch.linalg.vector_norm(target_obs, dim=-1).cpu().numpy()).statistic
    epe = torch.linalg.vector_norm(pred_obs - target_obs, dim=-1).mean()
    zero_epe = target_mag[valid_flat & (target_mag > 0.5)].mean()
    return {"observable_cells": int(observable.sum()), "median_angular_error_deg": float(angular.median()), "direction_sign_accuracy": float((cosine > 0).float().mean()), "magnitude_spearman": None if not np.isfinite(rho) else float(rho), "epe": float(epe), "zero_epe": float(zero_epe), "epe_improvement_vs_zero": float(1.0 - epe / zero_epe.clamp_min(1e-8))}


def _evaluate_field(model: torch.nn.Module, dataset: RealTextureRotationDataset, device: torch.device, *, collect_panel: bool) -> dict[str, object]:
    model.eval()
    rows = []
    metrics_by_polarity = {"on": [], "off": [], "combined": []}
    endpoint_errors, zero_endpoint_errors, solver_errors, zero_solver_errors = [], [], [], []
    panel = None
    for index in range(len(dataset)):
        item = dataset[index]
        frames = item["frames"].unsqueeze(0).to(device)
        with torch.inference_mode():
            output = model(frames)
            combined_energy = torch.cat((output["on_motion_energy"], output["off_motion_energy"]), dim=2)
            decoded = decode_direction_population(combined_energy, tuple(model.scales), field_size=8, direction_order="canonical")
        target_flow = item["dense_flow"].unsqueeze(0).to(device)
        target_mask = item["observability_mask"].unsqueeze(0).to(device)
        target_field, target_mask = _mean_pool(target_flow, target_mask, (8, 8))
        on_velocity, off_velocity = decoded["velocity"][:, :, 0], decoded["velocity"][:, :, 1]
        on_conf, off_conf = decoded["confidence"][:, :, 0], decoded["confidence"][:, :, 1]
        combined_velocity = (on_velocity * on_conf.unsqueeze(2) + off_velocity * off_conf.unsqueeze(2)) / (on_conf + off_conf).unsqueeze(2).clamp_min(1e-8)
        row_metrics = {}
        for name, velocity in (("on", on_velocity), ("off", off_velocity), ("combined", combined_velocity)):
            row_metrics[name] = _field_metrics(velocity, target_field, target_mask)
            metrics_by_polarity[name].append(row_metrics[name])
        weights = target_mask[:, :, 0] * (on_conf + off_conf)
        solver_field = combined_velocity.clone()
        solver_field[:, :, 0] *= 8.0 / float(target_flow.shape[-1])
        solver_field[:, :, 1] *= 8.0 / float(target_flow.shape[-2])
        solver = solve_weighted_rotation(solver_field, weights, tartanair_v2_lcam_front_intrinsics(640, 640).resized(8, 8))
        target_steps = item["target_step_rotation_vectors"].unsqueeze(0).to(device)
        target_endpoint = item["target_rotation_vector"].unsqueeze(0).to(device)
        endpoint_prediction = output["rotation_vector"][:, -1]
        endpoint_errors.extend(torch.rad2deg(torch.linalg.vector_norm(endpoint_prediction - target_endpoint, dim=-1)).cpu().tolist())
        zero_endpoint_errors.extend(torch.rad2deg(torch.linalg.vector_norm(target_endpoint, dim=-1)).cpu().tolist())
        solver_errors.extend(torch.rad2deg(torch.linalg.vector_norm(solver["rotation_vector"] - target_steps, dim=-1)).cpu().reshape(-1).tolist())
        zero_solver_errors.extend(torch.rad2deg(torch.linalg.vector_norm(target_steps, dim=-1)).cpu().reshape(-1).tolist())
        rows.append({"metadata": item["metadata"], "polarity": row_metrics, "solver_condition_median": float(solver["condition"].median()), "solver_valid_fraction": float(solver["valid"].float().mean())})
        if collect_panel and panel is None:
            panel = {"frame": item["frames"][0].permute(1, 2, 0).numpy(), "flow": target_field[0, 0].permute(1, 2, 0).cpu().numpy(), "prediction": combined_velocity[0, 0].permute(1, 2, 0).cpu().numpy(), "mask": target_mask[0, 0, 0].cpu().numpy()}

    def aggregate(values: list[dict[str, float | int | None]]) -> dict[str, object]:
        keys = ("observable_cells", "median_angular_error_deg", "direction_sign_accuracy", "magnitude_spearman", "epe", "zero_epe", "epe_improvement_vs_zero")
        return {key: float(np.nanmean([float(item[key]) for item in values if item[key] is not None])) if any(item[key] is not None for item in values) else None for key in keys} | {"samples": len(values)}

    return {"samples": len(dataset), "polarity": {key: aggregate(value) for key, value in metrics_by_polarity.items()}, "learned_endpoint_geodesic_deg": float(np.mean(endpoint_errors)), "zero_endpoint_geodesic_deg": float(np.mean(zero_endpoint_errors)), "analytic_solver_geodesic_deg": float(np.mean(solver_errors)), "analytic_solver_zero_geodesic_deg": float(np.mean(zero_solver_errors)), "rows": rows, "panel": panel}


def _save_quiver(panel: dict[str, np.ndarray], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)
    for axis, key, title in ((axes[0], "flow", "ground truth"), (axes[1], "prediction", "canonical decoded")):
        field = panel[key]
        axis.imshow(panel["frame"], alpha=0.55)
        yy, xx = np.mgrid[0 : field.shape[0], 0 : field.shape[1]]
        axis.quiver(xx, yy, field[..., 0], field[..., 1], color="yellow", angles="xy", scale_units="xy", scale=1)
        axis.set_title(title)
        axis.set_axis_off()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _run_field(output: Path, index_path: Path, checkpoint: Path, device: torch.device, limit: int, seed: int) -> dict[str, object]:
    records = json.loads(index_path.read_text(encoding="utf-8"))
    splits = default_split(records)
    source_manifest = {split: [path_free_record(record) for record in values] for split, values in splits.items()}
    report: dict[str, object] = {"experiment": {"name": "corrected-real-texture-controlled-geometry", "seed": seed, "image_size": [128, 128], "windows": [3, 5, 7], "canonical_direction_order": list(CANONICAL_DIRECTION_NAMES), "source_storage": "operator supplied fast-cache RGB only", "test_selection": "disabled"}, "commit": _commit(), "source_split_manifest_sha256": manifest_sha256(source_manifest), "checkpoint_sha256": _sha256(checkpoint), "splits": {}}
    model = FlyRotV1(scales=(4, 8, 16)).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model"] if "model" in state else state)
    for window in (3, 5, 7):
        report["splits"][str(window)] = {}
        for split_name in ("train", "validation", "test"):
            dataset = RealTextureRotationDataset(splits[split_name], window_length=window, image_size=(128, 128), limit=limit, seed=seed)
            result = _evaluate_field(model, dataset, device, collect_panel=split_name == "validation" and window == 3)
            panel = result.pop("panel", None)
            report["splits"][str(window)][split_name] = {"dataset_manifest": dataset.manifest(), "dataset_manifest_sha256": manifest_sha256(dataset.manifest()), "v1_corrected": result}
            if panel:
                _save_quiver(panel, output / "corrected_field_quiver_panel.png")
    (output / "corrected_real_texture_field.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _copy_historical(output: Path, path: Path, name: str) -> None:
    if path.is_file():
        shutil.copyfile(path, output / name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--index", type=Path, default=Path("artifacts/tartanair2_kiousb_lcam_index.json"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--historical-current", type=Path, required=True)
    parser.add_argument("--historical-small", type=Path, required=True)
    parser.add_argument("--device", default="cuda:1" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--field-limit", type=int, default=48)
    parser.add_argument("--seed", type=int, default=20260823)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"requested {device}, but CUDA is unavailable")

    _copy_historical(args.output_dir, args.historical_current, "t4t5_current_cases.json")
    _copy_historical(args.output_dir, args.historical_small, "t4t5_small_cases.json")
    full = _run_t4_grid(args.output_dir, (4, 8, 16), device, small=False)
    small = _run_t4_grid(args.output_dir, (1, 2, 4, 8), device, small=True)
    polarity = _pure_polarity_audit(args.output_dir, device, (4, 8, 16))
    magnitude = _scale_audit(args.output_dir, device)
    historical_rows = json.loads((args.output_dir / "t4t5_current_cases.json").read_text(encoding="utf-8")).get("cases", [])
    historical_rows = [row for row in historical_rows if _is_motion(str(row.get("kind"))) and row.get("contrast") == "medium" and row.get("displacement") == 4.0 and row.get("gamma") == 1.0 and row.get("noise_std") == 0.0 and row.get("blur_sigma") == 0.0 and row.get("brightness_offset") == 0.0 and row.get("valid_fraction", 0.0) > 0]
    legacy_response = torch.tensor([row["response"] for row in historical_rows], device=device)
    expected = torch.tensor([direction_index(row["angle_deg"]) for row in historical_rows], device=device)
    d4 = {"historical_legacy_as_canonical": d4_audit(legacy_response, expected), "historical_explicit_legacy_to_canonical": d4_audit(reorder_direction_vector(legacy_response), expected), "contract": {"legacy_order": ["E", "NE", "N", "NW", "W", "SW", "S", "SE"], "canonical_order": list(CANONICAL_DIRECTION_NAMES), "selection_split": "validation only; no test selection"}}
    (args.output_dir / "direction_mapping_audit.json").write_text(json.dumps(_json(d4), indent=2), encoding="utf-8")
    field = _run_field(args.output_dir, args.index, args.checkpoint, device, args.field_limit, args.seed)
    combined = {"commit": _commit(), "device": str(device), "t4t5_full_cases": len(full["cases"]), "t4t5_small_cases": len(small["cases"]), "field": field["experiment"], "checkpoint_sha256": field["checkpoint_sha256"], "polarity_variants": list(polarity["variants"]), "magnitude_status": magnitude["physical_pixel_calibration"], "d4_best_after_explicit_conversion": d4["historical_explicit_legacy_to_canonical"]["best_physical_mapping"]}
    (args.output_dir / "audit_run_manifest.json").write_text(json.dumps(combined, indent=2), encoding="utf-8")
    print(json.dumps(combined, indent=2))


if __name__ == "__main__":
    main()
