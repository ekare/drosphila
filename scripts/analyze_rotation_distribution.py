#!/usr/bin/env python3
"""Summarize relative rotation and translation magnitudes for one pose file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from flyrot.geometry.pose_conventions import load_tartanair_poses, relative_camera_rotation_vectors


def summary(values: np.ndarray) -> dict[str, float]:
    return {
        "min": float(np.min(values)),
        "p25": float(np.percentile(values, 25)),
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "p75": float(np.percentile(values, 75)),
        "p90": float(np.percentile(values, 90)),
        "max": float(np.max(values)),
    }


def analyze(path: Path) -> dict:
    poses = load_tartanair_poses(path)
    rotation_vectors = relative_camera_rotation_vectors(poses)
    angles = np.linalg.norm(rotation_vectors, axis=1)
    translations = np.linalg.norm(np.diff(poses[:, :3], axis=0), axis=1)
    return {
        "pose_path": str(path),
        "pose_count": int(len(poses)),
        "relative_count": int(len(rotation_vectors)),
        "quaternion_norm_max_error": float(
            np.max(np.abs(np.linalg.norm(poses[:, 3:7], axis=1) - 1.0))
        ),
        "rotation_angle_deg": summary(np.rad2deg(angles)),
        "rotation_vector_rad": {
            axis: summary(rotation_vectors[:, index])
            for index, axis in enumerate(("rx", "ry", "rz"))
        },
        "translation_magnitude": summary(translations),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pose", type=Path)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()

    result = analyze(args.pose)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    angle = result["rotation_angle_deg"]
    translation = result["translation_magnitude"]
    lines = [
        "# Rotation distribution",
        "",
        f"Pose source: `{result['pose_path']}`",
        "",
        f"- Pose count: `{result['pose_count']}`",
        f"- Relative pairs: `{result['relative_count']}`",
        f"- Maximum quaternion norm error: `{result['quaternion_norm_max_error']:.3e}`",
        "",
        "## Relative angle (degrees)",
        "",
        "| min | p25 | median | mean | p75 | p90 | max |",
        "|---:|---:|---:|---:|---:|---:|---:|",
        "| " + " | ".join(f"{angle[key]:.6f}" for key in ("min", "p25", "median", "mean", "p75", "p90", "max")) + " |",
        "",
        "## Translation magnitude per pair",
        "",
        "| min | p25 | median | mean | p75 | p90 | max |",
        "|---:|---:|---:|---:|---:|---:|---:|",
        "| " + " | ".join(f"{translation[key]:.6f}" for key in ("min", "p25", "median", "mean", "p75", "p90", "max")) + " |",
        "",
        "This report validates numeric pose parsing and relative-rotation label generation. It does not yet validate image-motion sign independently.",
        "",
    ]
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
