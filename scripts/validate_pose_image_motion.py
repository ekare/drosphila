#!/usr/bin/env python3
"""Compare pose-implied rotational image flow with real LK feature flow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from flyrot.geometry.pose_conventions import load_tartanair_poses, relative_camera_rotations


def rotational_flow(rotation: np.ndarray, points: np.ndarray, focal: float, center: np.ndarray) -> np.ndarray:
    rays = np.column_stack(((points - center) / focal, np.ones(len(points))))
    rotated = rays @ rotation.T
    projected = rotated[:, :2] / rotated[:, 2:3]
    return projected * focal + center - points


def cosine_scores(observed: np.ndarray, predicted: np.ndarray) -> tuple[float, float, int]:
    observed_norm = np.linalg.norm(observed, axis=1)
    predicted_norm = np.linalg.norm(predicted, axis=1)
    valid = (observed_norm > 0.5) & (predicted_norm > 1e-4) & np.isfinite(predicted).all(axis=1)
    if not np.any(valid):
        return float("nan"), float("nan"), 0
    cosine = np.sum(observed[valid] * predicted[valid], axis=1) / (
        observed_norm[valid] * predicted_norm[valid]
    )
    return float(np.median(cosine)), float(np.mean(cosine > 0)), int(valid.sum())


def inspect_pair(image_a: Path, image_b: Path, poses: np.ndarray, index: int, focal: float) -> dict:
    first = cv2.imread(str(image_a), cv2.IMREAD_GRAYSCALE)
    second = cv2.imread(str(image_b), cv2.IMREAD_GRAYSCALE)
    if first is None or second is None:
        raise RuntimeError(f"could not read image pair: {image_a}, {image_b}")
    points = cv2.goodFeaturesToTrack(first, maxCorners=1000, qualityLevel=0.01, minDistance=5)
    if points is None:
        return {"pair": index, "tracked": 0}
    next_points, status, _ = cv2.calcOpticalFlowPyrLK(first, second, points, None)
    status = status.reshape(-1).astype(bool)
    old = points.reshape(-1, 2)[status]
    new = next_points.reshape(-1, 2)[status]
    observed = new - old
    center = np.array([first.shape[1] / 2.0, first.shape[0] / 2.0])
    relative = relative_camera_rotations(poses[index : index + 2])[0]
    forward = rotational_flow(relative.T, old, focal, center)
    inverse = rotational_flow(relative, old, focal, center)
    forward_score = cosine_scores(observed, forward)
    inverse_score = cosine_scores(observed, inverse)
    return {
        "pair": index,
        "tracked": int(len(old)),
        "observed_flow_median_px": float(np.median(np.linalg.norm(observed, axis=1))),
        "pose_relative_angle_deg": float(np.rad2deg(np.linalg.norm(np.arccos(np.clip((np.trace(relative) - 1) / 2, -1, 1))))),
        "candidate_r_rel_transpose": {
            "median_cosine": forward_score[0],
            "positive_fraction": forward_score[1],
            "valid": forward_score[2],
        },
        "candidate_r_rel": {
            "median_cosine": inverse_score[0],
            "positive_fraction": inverse_score[1],
            "valid": inverse_score[2],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sequence", type=Path)
    parser.add_argument("--pose", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    parser.add_argument("--focal", type=float, default=320.0)
    parser.add_argument("--pairs", type=int, nargs="+", default=[0, 10, 50, 100])
    args = parser.parse_args()

    poses = load_tartanair_poses(args.pose)
    image_paths = sorted(args.sequence.glob("*.png"))
    if len(image_paths) != len(poses):
        raise RuntimeError(f"image/pose count mismatch: {len(image_paths)} vs {len(poses)}")
    results = []
    for index in args.pairs:
        if index < 0 or index + 1 >= len(image_paths):
            continue
        results.append(inspect_pair(image_paths[index], image_paths[index + 1], poses, index, args.focal))

    output = {
        "sequence": str(args.sequence),
        "pose": str(args.pose),
        "focal_px": args.focal,
        "center_policy": "image center",
        "pairs": results,
        "interpretation": "R_rel.T maps current camera rays to next-camera rays for pure rotation under T_world_camera.",
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Pose/image-motion validation",
        "",
        f"Sequence: `{args.sequence}`",
        f"Focal assumption: `{args.focal:.1f}` px; principal point: image center",
        "",
        "The pose-derived candidate is compared with real Lucas–Kanade feature flow. Translation and scene depth are not removed, so this is a directional sanity check, not a complete calibration proof.",
        "",
        "| pair | tracked | angle deg | median flow px | R_rel.T cosine | R_rel cosine |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        lines.append(
            f"| {result['pair']} | {result['tracked']} | {result['pose_relative_angle_deg']:.3f} | "
            f"{result['observed_flow_median_px']:.3f} | "
            f"{result['candidate_r_rel_transpose']['median_cosine']:.3f} | "
            f"{result['candidate_r_rel']['median_cosine']:.3f} |"
        )
    lines.append("")
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
