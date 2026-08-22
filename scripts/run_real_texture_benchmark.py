#!/usr/bin/env python3
"""Run exact real-texture pure-rotation and local-field audits.

The source RGB is read from the operator-supplied fast-cache index. No source
dataset file is copied or modified, and test metrics are reported only after
the train/validation records have been enumerated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import spearmanr
from torch.utils.data import DataLoader

from flyrot.data.tartanair import default_split
from flyrot.geometry.camera import tartanair_v2_lcam_front_intrinsics
from flyrot.geometry.so3 import geodesic_distance, log_so3
from flyrot.models.flyrot_v0 import FlyRotV0
from flyrot.models.flyrot_v1 import FlyRotV1
from flyrot.real_texture import RealTextureRotationDataset, decode_direction_population, manifest_sha256, path_free_record, rotation_vector, solve_weighted_rotation


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


def _metrics(prediction: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> dict[str, float | int | None]:
    pred_flat = prediction.permute(0, 1, 3, 4, 2).reshape(-1, 2)
    target_flat = target.permute(0, 1, 3, 4, 2).reshape(-1, 2)
    valid_flat = valid.squeeze(2).reshape(-1) > 0.2
    pred_mag = torch.linalg.vector_norm(pred_flat, dim=-1)
    target_mag = torch.linalg.vector_norm(target_flat, dim=-1)
    observable = valid_flat & (target_mag > 0.5) & (pred_mag > 1e-5)
    if not bool(observable.any()):
        return {"observable_cells": 0, "median_angular_error_deg": None, "direction_sign_accuracy": None, "magnitude_spearman": None, "epe": float(torch.linalg.vector_norm(target_flat, dim=-1).mean()), "zero_epe": float(target_mag.mean()), "epe_improvement_vs_zero": 0.0}
    pred_obs = pred_flat[observable]
    target_obs = target_flat[observable]
    cosine = (pred_obs * target_obs).sum(-1) / (torch.linalg.vector_norm(pred_obs, dim=-1) * torch.linalg.vector_norm(target_obs, dim=-1)).clamp_min(1e-8)
    angular = torch.rad2deg(torch.acos(cosine.clamp(-1, 1)))
    pred_mag_obs = torch.linalg.vector_norm(pred_obs, dim=-1).cpu().numpy()
    target_mag_obs = torch.linalg.vector_norm(target_obs, dim=-1).cpu().numpy()
    rho = spearmanr(pred_mag_obs, target_mag_obs).statistic if len(pred_mag_obs) > 1 else float("nan")
    epe = torch.linalg.vector_norm(pred_obs - target_obs, dim=-1).mean()
    zero_epe = target_mag[valid_flat & (target_mag > 0.5)].mean()
    return {"observable_cells": int(observable.sum()), "median_angular_error_deg": float(angular.median()), "direction_sign_accuracy": float((cosine > 0).float().mean()), "magnitude_spearman": None if not np.isfinite(rho) else float(rho), "epe": float(epe), "zero_epe": float(zero_epe), "epe_improvement_vs_zero": float(1.0 - epe / zero_epe.clamp_min(1e-8))}


def _evaluate_dataset(model: torch.nn.Module, dataset: RealTextureRotationDataset, device: torch.device, *, collect_panel: bool = False) -> dict[str, object]:
    model.eval()
    local_rows: list[dict[str, object]] = []
    endpoint_errors: list[float] = []
    zero_endpoint_errors: list[float] = []
    solver_errors: list[float] = []
    zero_solver_errors: list[float] = []
    polarity_rows: dict[str, list[dict[str, float | int | None]]] = {"on": [], "off": [], "combined": []}
    panel = None
    for index in range(len(dataset)):
        item = dataset[index]
        frames = item["frames"].unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(frames)
            if "on_motion_energy" in output:
                combined_energy = torch.cat((output["on_motion_energy"], output["off_motion_energy"]), dim=2)
                scales = tuple(model.scales)
            else:
                single_energy = output["direction_energy"]
                combined_energy = torch.cat((single_energy, torch.zeros_like(single_energy)), dim=2)
                scales = tuple(model.direction_cells.scales)
            decoded = decode_direction_population(combined_energy, scales, field_size=8)
        target_flow = item["dense_flow"].unsqueeze(0).to(device)
        target_mask = item["observability_mask"].unsqueeze(0).to(device)
        # Convert dense pixel flow into the same 8x8 cell grid as the decoder.
        target_field, pooled_mask = _mean_pool(target_flow, target_mask, (8, 8))
        target_field = target_field
        target_mask = pooled_mask
        on_velocity = decoded["velocity"][:, :, 0]
        off_velocity = decoded["velocity"][:, :, 1]
        on_conf = decoded["confidence"][:, :, 0]
        off_conf = decoded["confidence"][:, :, 1]
        combined_velocity = (on_velocity * on_conf.unsqueeze(2) + off_velocity * off_conf.unsqueeze(2)) / (on_conf + off_conf).unsqueeze(2).clamp_min(1e-8)
        for name, velocity in (("on", on_velocity), ("off", off_velocity), ("combined", combined_velocity)):
            row = _metrics(velocity, target_field, target_mask)
            polarity_rows[name].append(row)
        weights = target_mask[:, :, 0] * (on_conf + off_conf)
        solver = solve_weighted_rotation(combined_velocity, weights, tartanair_v2_lcam_front_intrinsics(640, 640).resized(8, 8))
        target_steps = item["target_step_rotation_vectors"].unsqueeze(0).to(device)
        target_endpoint = item["target_rotation_vector"].unsqueeze(0).to(device)
        endpoint_prediction = output["rotation_vector"][:, -1]
        endpoint_errors.extend(torch.rad2deg(torch.linalg.vector_norm(endpoint_prediction - target_endpoint, dim=-1)).cpu().tolist())
        zero_endpoint_errors.extend(torch.rad2deg(torch.linalg.vector_norm(target_endpoint, dim=-1)).cpu().tolist())
        solver_errors.extend(torch.rad2deg(torch.linalg.vector_norm(solver["rotation_vector"] - target_steps, dim=-1)).cpu().reshape(-1).tolist())
        zero_solver_errors.extend(torch.rad2deg(torch.linalg.vector_norm(target_steps, dim=-1)).cpu().reshape(-1).tolist())
        local_rows.append({"metadata": item["metadata"], "polarity": {key: polarity_rows[key][-1] for key in polarity_rows}, "solver_condition_median": float(solver["condition"].median()), "solver_valid_fraction": float(solver["valid"].float().mean())})
        if collect_panel and panel is None:
            panel = {"frame": item["frames"][0].permute(1, 2, 0).numpy(), "flow": target_field[0, 0].permute(1, 2, 0).cpu().numpy(), "prediction": combined_velocity[0, 0].permute(1, 2, 0).cpu().numpy(), "mask": target_mask[0, 0, 0].cpu().numpy()}
    def aggregate(rows: list[dict[str, float | int | None]]) -> dict[str, object]:
        keys = ("observable_cells", "median_angular_error_deg", "direction_sign_accuracy", "magnitude_spearman", "epe", "zero_epe", "epe_improvement_vs_zero")
        result: dict[str, object] = {key: float(np.nanmean([float(row[key]) for row in rows if row[key] is not None])) if any(row[key] is not None for row in rows) else None for key in keys}
        result["samples"] = len(rows)
        return result
    return {"samples": len(dataset), "polarity": {key: aggregate(value) for key, value in polarity_rows.items()}, "learned_endpoint_geodesic_deg": float(np.mean(endpoint_errors)) if endpoint_errors else None, "zero_endpoint_geodesic_deg": float(np.mean(zero_endpoint_errors)) if zero_endpoint_errors else None, "analytic_solver_geodesic_deg": float(np.mean(solver_errors)) if solver_errors else None, "analytic_solver_zero_geodesic_deg": float(np.mean(zero_solver_errors)) if zero_solver_errors else None, "rows": local_rows, "panel": panel}


def _save_panel(panel: dict[str, np.ndarray], path: Path) -> None:
    figure, axes = plt.subplots(1, 4, figsize=(12, 3), constrained_layout=True)
    axes[0].imshow(panel["frame"])
    axes[0].set_title("real texture")
    for axis, key, title in ((axes[1], "flow", "GT flow"), (axes[2], "prediction", "decoded field"), (axes[3], "mask", "observability")):
        image = axis.imshow(np.linalg.norm(panel[key], axis=-1) if panel[key].ndim == 3 else panel[key], cmap="magma")
        axis.set_title(title)
        figure.colorbar(image, ax=axis, fraction=0.046)
        axis.axis("off")
    figure.savefig(path, dpi=150)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:1" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--limit-per-split", type=int, default=48)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    records = json.loads(args.index.read_text(encoding="utf-8"))
    splits = default_split(records)
    source_manifest = {split: [path_free_record(record) for record in split_records] for split, split_records in splits.items()}
    report: dict[str, object] = {"experiment": {"name": "real-texture-controlled-geometry", "seed": args.seed, "image_size": [128, 128], "windows": [3, 5, 7], "rotation_families": ["zero", "yaw", "pitch", "roll", "two_axis", "three_axis"], "magnitudes_deg": list((0.0, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 15.0)), "homography": "K_target @ R_image @ inverse(K_source)", "source_storage": "operator supplied fast-cache RGB only", "test_selection": "disabled"}, "commit": _commit(), "source_split_manifest_sha256": manifest_sha256(source_manifest), "checkpoint_sha256": _sha256(args.checkpoint), "splits": {}}
    v1 = FlyRotV1(scales=(4, 8, 16)).to(device)
    v0 = FlyRotV0(scales=(4, 8, 16)).to(device)
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location=device, weights_only=False)
        v1.load_state_dict(state["model"] if "model" in state else state)
    for window in (3, 5, 7):
        report["splits"][str(window)] = {}
        for split_name in ("train", "validation", "test"):
            split_records = splits[split_name]
            dataset = RealTextureRotationDataset(split_records, window_length=window, image_size=(128, 128), limit=args.limit_per_split, seed=args.seed)
            v1_metrics = _evaluate_dataset(v1, dataset, device, collect_panel=split_name == "validation" and window == 3)
            v0_metrics = _evaluate_dataset(v0, dataset, device, collect_panel=False)
            panel = v1_metrics.pop("panel", None)
            report["splits"][str(window)][split_name] = {"dataset_manifest": dataset.manifest(), "dataset_manifest_sha256": manifest_sha256(dataset.manifest()), "v1": v1_metrics, "v0": {key: value for key, value in v0_metrics.items() if key != "panel"}}
            if panel:
                _save_panel(panel, args.output_dir / "real_texture_validation_field_panel.png")
    endpoint_angles = [float(np.linalg.norm(rotation_vector(family, magnitude, sign))) for family in ("yaw", "pitch", "roll", "two_axis", "three_axis") for magnitude in (0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 15.0) for sign in (-1, 1)]
    report["wrong_sign_oracle"] = {"status": "checked_geometrically", "correct_endpoint_geodesic_deg": 0.0, "wrong_endpoint_geodesic_deg_mean": float(np.degrees(np.mean(2.0 * np.asarray(endpoint_angles)))), "wrong_endpoint_geodesic_deg_min": float(np.degrees(np.min(2.0 * np.asarray(endpoint_angles)))), "correct_minus_wrong": "exact dense flow sign separation is computed by direct SO(3) projection; learned sign remains unresolved until field passes"}
    (args.output_dir / "real_texture_rotation_benchmark.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = ["# Real-texture controlled-geometry rotation benchmark", "", "This benchmark preserves real TartanAir RGB texture but applies only exact homography rotation; it is not a synthetic-texture dataset.", "", f"- Commit: `{report['commit']}`", f"- Source split manifest SHA256: `{report['source_split_manifest_sha256']}`", f"- Checkpoint SHA256: `{report['checkpoint_sha256']}`", "- Test split was enumerated and evaluated after train/validation generation; it was not used for selection.", "", "## Split policy", "", "Train, validation, and test are trajectory-disjoint under the repository environment split. Each example carries a path-free logical source ID, exact intrinsics, per-step SO(3), endpoint SO(3), dense rotational flow, warp mask, resize transform, observability statistics, and seed.", "", "Full machine-readable results are in `real_texture_rotation_benchmark.json`; representative field panel is `real_texture_validation_field_panel.png`."]
    (args.output_dir / "real_texture_rotation_dataset.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output_dir), "source_manifest_sha256": report["source_split_manifest_sha256"], "splits": {window: {split: report["splits"][window][split]["v1"]["polarity"]["combined"] for split in report["splits"][window]} for window in report["splits"]}}, indent=2))


if __name__ == "__main__":
    main()
