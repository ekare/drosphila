#!/usr/bin/env python3
"""Run the bounded deterministic T4/T5-inspired cell validation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from flyrot.t4t5 import (
    ANGLES_DEG,
    CONTRASTS,
    DISPLACEMENTS,
    STIMULUS_CLASSES,
    StimulusSpec,
    direction_accuracy,
    evaluate_bank,
    make_stimulus,
    tensor_to_list,
)


def _git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _hash_json(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _is_motion(kind: str) -> bool:
    return kind.startswith("moving_") or kind in {"contrast_reversed_edge", "aperture_limited_edge"}


def _case(spec: StimulusSpec, scales: tuple[int, ...], device: torch.device) -> dict[str, object]:
    frames, valid = make_stimulus(spec)
    result = evaluate_bank(frames, valid, scales=scales, device=device)
    return {
        "kind": spec.kind,
        "angle_deg": spec.angle_deg,
        "displacement": spec.displacement,
        "contrast": spec.contrast_name,
        "gamma": spec.gamma,
        "brightness_offset": spec.brightness_offset,
        "noise_std": spec.noise_std,
        "blur_sigma": spec.blur_sigma,
        "direction_accuracy": direction_accuracy(spec, result),
        **{key: tensor_to_list(value) for key, value in result.items()},
    }


def _summarize(rows: list[dict[str, object]], polarity: str) -> dict[str, object]:
    motion = [row for row in rows if _is_motion(str(row["kind"])) and row["contrast"] in {"medium", "high"}]
    ideal_medium = [row for row in rows if _is_motion(str(row["kind"])) and row["contrast"] in {"medium", "high"} and row["gamma"] == 1.0 and row["noise_std"] == 0.0 and row["blur_sigma"] == 0.0 and row["brightness_offset"] == 0.0 and float(row["valid_fraction"]) > 0.0]
    responses = [float(row[f"{polarity}_response"][int(row["angle_deg"]) // 45]) for row in ideal_medium]
    ratios = [float(row["preferred_null_ratio"]) for row in ideal_medium if math.isfinite(float(row["preferred_null_ratio"]))]
    dsis = [float(row["dsi"]) for row in ideal_medium]
    accuracy = [float(row["direction_accuracy"]) for row in ideal_medium]
    return {
        "cases": len(motion),
        "ideal_medium_cases": len(ideal_medium),
        "direction_sign_accuracy": float(np.mean(accuracy)) if accuracy else None,
        "median_preferred_null_ratio": float(np.median(ratios)) if ratios else None,
        "median_dsi": float(np.median(dsis)) if dsis else None,
        "matched_motion_response_median": float(np.median(responses)) if responses else None,
    }


def _panel(output: Path, rows: list[dict[str, object]]) -> None:
    examples = [
        ("moving_bright_edge", 0, "bright_edge_east"),
        ("moving_dark_edge", 90, "dark_edge_south"),
        ("moving_bright_bar", 225, "bright_bar_sw"),
        ("flicker", 0, "flicker_false_positive"),
    ]
    figure, axes = plt.subplots(2, 2, figsize=(9, 7), constrained_layout=True)
    for axis, (kind, angle, title) in zip(axes.flat, examples):
        matches = [row for row in rows if row["kind"] == kind and row["angle_deg"] == angle and row["contrast"] == "medium" and row["displacement"] == 4.0]
        row = matches[0] if matches else None
        if row is None:
            axis.axis("off")
            continue
        values = np.asarray(row["response"], dtype=float)
        axis.bar(np.arange(8), values)
        axis.set_xticks(np.arange(8), ("E", "NE", "N", "NW", "W", "SW", "S", "SE"))
        axis.set_title(f"{title}; pref={row['preferred_index']}")
        axis.set_ylabel("mean cell response")
    figure.savefig(output, dpi=140)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scales", default="4,8,16")
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--device", default="cuda:1" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scales = tuple(int(value) for value in args.scales.split(","))
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"requested {device}, but CUDA is unavailable")
    rows: list[dict[str, object]] = []
    # Full requested grid plus controlled nuisance variants.  The grid is
    # deterministic and intentionally bounded; no hyperparameter search occurs.
    for kind in STIMULUS_CLASSES:
        angles = ANGLES_DEG if _is_motion(kind) else (0,)
        displacements = DISPLACEMENTS if _is_motion(kind) else (0.0,)
        contrasts = tuple(CONTRASTS) if kind not in {"stationary_uniform", "brightness_increase", "brightness_decrease", "flicker"} else ("medium",)
        for angle in angles:
            for displacement in displacements:
                for contrast in contrasts:
                    rows.append(_case(StimulusSpec(kind=kind, angle_deg=angle, displacement=displacement, contrast_name=contrast, height=128, width=160, seed=args.seed), scales, device))
    nuisance = [
        StimulusSpec("moving_bright_edge", 0, 4, "medium", height=128, width=160, gamma=1.8, seed=args.seed),
        StimulusSpec("moving_bright_edge", 45, 4, "medium", height=128, width=160, brightness_offset=0.08, seed=args.seed),
        StimulusSpec("moving_dark_edge", 180, 4, "medium", height=128, width=160, noise_std=0.02, seed=args.seed),
        StimulusSpec("moving_bright_bar", 270, 4, "medium", height=128, width=160, blur_sigma=0.8, seed=args.seed),
        StimulusSpec("moving_bright_edge", 0, 16, "medium", width=192, height=128, seed=args.seed),
    ]
    rows.extend(_case(spec, scales, device) for spec in nuisance)
    summaries = {polarity: _summarize(rows, polarity) for polarity in ("on", "off")}
    motion_rows = [row for row in rows if _is_motion(str(row["kind"])) and row["contrast"] == "medium" and row["displacement"] in {4.0, 8.0}]
    matched = float(np.median([float(row["preferred_response"]) for row in motion_rows]))
    flicker = [row for row in rows if row["kind"] == "flicker"]
    stationary = [row for row in rows if row["kind"] == "stationary_uniform"]
    crosstalk = []
    for row in motion_rows:
        expected = "on" if str(row["kind"]) in {"moving_bright_edge", "moving_bright_bar", "moving_patch", "aperture_limited_edge"} else "off"
        other = "off" if expected == "on" else "on"
        crosstalk.append(float(np.max(row[f"{other}_response"])) / max(float(np.max(row[f"{expected}_response"])), 1e-8))
    summary = {
        "t4_on": summaries["on"],
        "t5_off": summaries["off"],
        "matched_motion_response_median": matched,
        "flicker_false_positive_ratio": float(np.median([float(row["preferred_response"]) for row in flicker])) / max(matched, 1e-8),
        "stationary_false_positive_ratio": float(np.median([float(row["preferred_response"]) for row in stationary])) / max(matched, 1e-8),
        "polarity_crosstalk_median": float(np.median(crosstalk)) if crosstalk else None,
        "c0": {
            "status": "passed" if summaries["on"]["direction_sign_accuracy"] >= 0.95 and summaries["off"]["direction_sign_accuracy"] >= 0.95 and np.median([float(row["preferred_null_ratio"]) for row in motion_rows]) >= 3.0 and np.median([float(row["dsi"]) for row in motion_rows]) >= 0.5 and float(np.median([float(row["preferred_response"]) for row in flicker])) <= matched * 0.1 and float(np.median([float(row["preferred_response"]) for row in stationary])) <= matched * 0.05 and (float(np.median(crosstalk)) if crosstalk else 1.0) <= 0.25 else "failed",
            "thresholds_are_engineering_only": True,
            "correction_rounds": ["none; current scales measured first", "small-motion candidate reserved if low-speed coverage fails"],
        },
    }
    manifest = {
        "name": "t4t5-inspired-cell-validation-v0.3.1",
        "seed": args.seed,
        "raster": [160, 128],
        "steps": 5,
        "scales": list(scales),
        "device": str(device),
        "stimulus_classes": list(STIMULUS_CLASSES),
        "angles_deg": list(ANGLES_DEG),
        "displacements": list(DISPLACEMENTS),
        "contrasts": list(CONTRASTS),
        "claim_boundary": "T4/T5-inspired engineering validation; not biological equivalence",
    }
    report = {"experiment": manifest, "dataset_manifest_sha256": _hash_json(manifest), "commit": _git_commit(), "checkpoint_sha256": None, "summary": summary, "cases": rows}
    (args.output_dir / "t4t5_cell_validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    _panel(args.output_dir / "t4t5_representative_panels.png", rows)
    md = [
        "# T4/T5-inspired cell validation",
        "",
        "This is an engineering validation of separated ON/T4-inspired and OFF/T5-inspired paths; it is not a biological-equivalence claim.",
        "",
        f"- Commit: `{report['commit']}`",
        f"- Seed: `{args.seed}`",
        f"- Dataset manifest SHA256: `{report['dataset_manifest_sha256']}`",
        f"- Checkpoint: `none` (fixed front-end cell bank)",
        f"- Raster: `160x128` non-square; current scales: `{scales}`",
        "",
        "## C0 result",
        "",
        f"- Status: **{summary['c0']['status']}**",
        f"- ON/T4-inspired: `{json.dumps(summaries['on'])}`",
        f"- OFF/T5-inspired: `{json.dumps(summaries['off'])}`",
        f"- Flicker false-positive ratio: `{summary['flicker_false_positive_ratio']:.5f}`",
        f"- Stationary false-positive ratio: `{summary['stationary_false_positive_ratio']:.5f}`",
        f"- Median polarity cross-talk: `{summary['polarity_crosstalk_median']}`",
        "",
        "## Bounded decision",
        "",
        "The full requested deterministic grid was run once. If C0 fails, only the explicitly named polarity/sign/timing audit and one small-motion scale candidate are allowed; no unbounded search is performed.",
        "",
        "Representative tuning panels: `t4t5_representative_panels.png`.",
    ]
    (args.output_dir / "t4t5_cell_validation.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({"status": summary["c0"]["status"], "summary": summary, "manifest_sha256": report["dataset_manifest_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
