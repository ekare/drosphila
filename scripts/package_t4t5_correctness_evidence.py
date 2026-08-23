#!/usr/bin/env python3
"""Build a path-scrubbed, source-free correctness evidence archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import tarfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from flyrot.validation_contract import population_decode, signed_angle_delta_deg


def _copy(source: Path, target: Path) -> None:
    if source.is_file():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)


def _errors(path: Path, *, converted: bool = False) -> list[float]:
    report = json.loads(path.read_text(encoding="utf-8"))
    values = []
    for row in report.get("cases", []):
        if not str(row.get("kind", "")).startswith("moving_"):
            continue
        if row.get("contrast") != "medium":
            continue
        response = torch.tensor(row["response"], dtype=torch.float32).unsqueeze(0)
        expected = torch.tensor([int(row.get("direction_index", round(float(row["angle_deg"]) / 45.0) % 8))])
        if converted:
            from flyrot.validation_contract import reorder_direction_vector

            response = reorder_direction_vector(response)
        decoded = population_decode(response)
        error = signed_angle_delta_deg(decoded["population_angle_deg"], expected.float() * 45.0)[0]
        if bool(decoded["population_valid"][0]):
            values.append(float(error))
    return values


def _confusion(path: Path) -> list[list[int]]:
    report = json.loads(path.read_text(encoding="utf-8"))
    matrix = np.zeros((8, 8), dtype=np.int64)
    for row in report.get("cases", []):
        if not str(row.get("kind", "")).startswith("moving_") or row.get("contrast") != "medium":
            continue
        expected = int(row.get("direction_index", round(float(row["angle_deg"]) / 45.0) % 8))
        predicted = int(row["preferred_index"])
        matrix[expected, predicted] += 1
    return matrix.tolist()


def _stimulus_strip(path: Path) -> None:
    from flyrot.validation_contract import pure_edge_stimulus

    fig, axes = plt.subplots(2, 5, figsize=(12, 5), constrained_layout=True)
    for row_index, kind in enumerate(("on", "off")):
        frames, _ = pure_edge_stimulus(kind, angle_deg=45, displacement=4, height=128, width=160)
        for frame_index, axis in enumerate(axes[row_index]):
            axis.imshow(frames[frame_index, 0], cmap="gray", vmin=0, vmax=1)
            axis.set_title(f"{kind} t={frame_index}")
            axis.axis("off")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--historical-real-texture", type=Path)
    args = parser.parse_args()
    staging = args.audit_dir / "evidence_bundle"
    staging.mkdir(parents=True, exist_ok=True)

    copies = {
        "t4t5_current_cases.json": "t4t5_current_cases.json",
        "t4t5_small_cases.json": "t4t5_small_cases.json",
        "t4t5_corrected_full_cases.json": "t4t5_corrected_full_cases.json",
        "t4t5_corrected_small_cases.json": "t4t5_corrected_small_cases.json",
        "polarity_frontend_audit.json": "polarity_frontend_audit.json",
        "direction_mapping_audit.json": "direction_mapping_audit.json",
        "magnitude_calibration_audit.json": "magnitude_calibration_audit.json",
        "corrected_real_texture_field.json": "corrected_real_texture_field.json",
        "corrected_field_quiver_panel.png": "corrected_field_quiver_panel.png",
    }
    for source_name, target_name in copies.items():
        _copy(args.audit_dir / source_name, staging / target_name)
    if args.historical_real_texture:
        _copy(args.historical_real_texture, staging / "real_texture_historical_full.json")
    for name in ("direction_mapping_audit.json", "polarity_frontend_audit.json", "magnitude_calibration_audit.json", "t4t5_validation_correctness_audit.json", "t4t5_validation_correctness_audit.md", "c0_f0_before_after.md"):
        _copy(args.repo / "reports" / name, staging / "reports" / name)

    corrected_full = args.audit_dir / "t4t5_corrected_full_cases.json"
    current = args.audit_dir / "t4t5_current_cases.json"
    signed = {"historical_legacy_as_canonical_deg": _errors(current), "historical_explicit_conversion_deg": _errors(current, converted=True), "corrected_full_deg": _errors(corrected_full)}
    (staging / "signed_error_histograms.json").write_text(json.dumps(signed, indent=2), encoding="utf-8")
    (staging / "confusion_matrices.json").write_text(json.dumps({"historical_current_argmax": _confusion(current), "corrected_full_argmax": _confusion(corrected_full)}, indent=2), encoding="utf-8")
    _stimulus_strip(staging / "stimulus_strips.png")

    commands = [
        "git rev-parse HEAD",
        "nvidia-smi --query-gpu=index,name,pci.bus_id,utilization.gpu,memory.used,memory.total --format=csv",
        "python scripts/run_t4t5_correctness_audit.py --output-dir NV1_AUDIT --device cuda:1",
        "python scripts/package_t4t5_correctness_evidence.py --audit-dir NV1_AUDIT --output NV1_BUNDLE",
    ]
    (staging / "commands.txt").write_text("\n".join(commands) + "\n", encoding="utf-8")
    (staging / "README.txt").write_text(
        "Source-free correctness evidence. No source RGB, checkpoint, secret, absolute path, or machine inventory is included. "
        "The historical JSON files are copied as evidence; they are not authoritative release artifacts. "
        "Canonical order is E,SE,S,SW,W,NW,N,NE with x-right/y-down image coordinates.\n",
        encoding="utf-8",
    )

    slash = chr(47)
    forbidden_tokens = [slash + "media" + slash, slash + "home" + slash, slash + "mnt" + slash, "emre" + "@", "hyper" + "ion"]
    forbidden = re.compile("|".join(re.escape(token) for token in forbidden_tokens) + r"|[A-Za-z]:\\\\|GPU-[0-9A-Fa-f-]{16,}|192\\.168\\.|10\\.\\d+\\.\\d+\\.\\d+")
    violations = []
    for file in staging.rglob("*"):
        if file.is_file() and file.suffix.lower() in {".json", ".md", ".txt"}:
            text = file.read_text(encoding="utf-8", errors="replace")
            if forbidden.search(text):
                violations.append(file.name)
    if violations:
        raise RuntimeError(f"privacy scan failed: {violations}")

    hashes = []
    for file in sorted(staging.rglob("*")):
        if file.is_file():
            digest = hashlib.sha256(file.read_bytes()).hexdigest()
            hashes.append(f"{digest}  {file.relative_to(staging)}")
    (staging / "hashes.txt").write_text("\n".join(hashes) + "\n", encoding="utf-8")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(args.output, "w:gz") as archive:
        for file in sorted(staging.rglob("*")):
            if file.is_file():
                archive.add(file, arcname=file.relative_to(staging))
    size = args.output.stat().st_size
    if size > 100 * 1024 * 1024:
        raise RuntimeError(f"bundle exceeds 100 MB: {size}")
    print(json.dumps({"output": str(args.output), "bytes": size, "files": len(hashes), "privacy_scan": "passed"}, indent=2))


if __name__ == "__main__":
    main()
