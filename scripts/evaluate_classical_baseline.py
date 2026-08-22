#!/usr/bin/env python3
"""Evaluate a calibrated OpenCV relative-rotation baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from flyrot.geometry.pose_conventions import load_tartanair_poses, relative_camera_rotations
from flyrot.geometry.so3 import geodesic_distance


def sorted_images(root: Path) -> list[Path]:
    return sorted(root.glob("*.png"), key=lambda path: int(path.name.split("_", 1)[0]))


def estimate_rotation(first: np.ndarray, second: np.ndarray, focal: float, center: tuple[float, float]) -> tuple[np.ndarray | None, int]:
    points = cv2.goodFeaturesToTrack(first, maxCorners=2000, qualityLevel=0.01, minDistance=5)
    if points is None:
        return None, 0
    tracked, status, _ = cv2.calcOpticalFlowPyrLK(first, second, points, None)
    valid = status.reshape(-1).astype(bool)
    p0 = points.reshape(-1, 2)[valid]
    p1 = tracked.reshape(-1, 2)[valid]
    if len(p0) < 8:
        return None, int(len(p0))
    camera = np.array([[focal, 0, center[0]], [0, focal, center[1]], [0, 0, 1]], dtype=np.float64)
    essential, mask = cv2.findEssentialMat(p0, p1, camera, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    if essential is None:
        return None, 0
    try:
        _, rotation, _, pose_mask = cv2.recoverPose(essential, p0, p1, camera, mask=mask)
    except cv2.error:
        return None, 0
    return rotation.astype(np.float64), int(pose_mask.sum()) if pose_mask is not None else 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sequence", type=Path)
    parser.add_argument("--pose", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    parser.add_argument("--focal", type=float, default=320.0)
    parser.add_argument("--max-pairs", type=int, default=100)
    parser.add_argument("--frame-gap", type=int, default=1)
    args = parser.parse_args()

    poses = load_tartanair_poses(args.pose)
    images = sorted_images(args.sequence)
    center = (images and cv2.imread(str(images[0]), cv2.IMREAD_GRAYSCALE).shape[1] / 2.0, images and cv2.imread(str(images[0]), cv2.IMREAD_GRAYSCALE).shape[0] / 2.0)
    errors = []
    zero_errors = []
    inliers = []
    attempted = min(args.max_pairs, len(images) - args.frame_gap, len(poses) - args.frame_gap)
    for index in range(attempted):
        first = cv2.imread(str(images[index]), cv2.IMREAD_GRAYSCALE)
        second = cv2.imread(str(images[index + args.frame_gap]), cv2.IMREAD_GRAYSCALE)
        estimated, inlier_count = estimate_rotation(first, second, args.focal, center)
        gt_relative = relative_camera_rotations(poses[[index, index + args.frame_gap]])[0]
        target = gt_relative.T
        zero_errors.append(float(np.rad2deg(geodesic_distance(np.eye(3), target))))
        if estimated is not None:
            errors.append(float(np.rad2deg(geodesic_distance(estimated, target))))
            inliers.append(inlier_count)

    result = {
        "sequence": str(args.sequence),
        "pose": str(args.pose),
        "focal_px": args.focal,
        "frame_gap": args.frame_gap,
        "attempted_pairs": attempted,
        "successful_pairs": len(errors),
        "zero_baseline_median_deg": float(np.median(zero_errors)) if zero_errors else None,
        "zero_baseline_mean_deg": float(np.mean(zero_errors)) if zero_errors else None,
        "classical_median_deg": float(np.median(errors)) if errors else None,
        "classical_mean_deg": float(np.mean(errors)) if errors else None,
        "classical_p90_deg": float(np.percentile(errors, 90)) if errors else None,
        "median_inliers": float(np.median(inliers)) if inliers else None,
        "target_convention": "OpenCV recoverPose R compared to R_rel.T, where R_rel=R_world_camera_t.T @ R_world_camera_t+1",
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Calibrated classical baseline",
        "",
        f"Sequence: `{args.sequence}`",
        f"Focal assumption: `{args.focal:.1f}` px; principal point: image center",
        f"Frame gap: `{args.frame_gap}`",
        f"Attempted pairs: `{attempted}`; successful recoveries: `{len(errors)}`",
        "",
        "| metric | degrees |",
        "|---|---:|",
        f"| zero baseline median | {result['zero_baseline_median_deg']} |",
        f"| zero baseline mean | {result['zero_baseline_mean_deg']} |",
        f"| classical median | {result['classical_median_deg']} |",
        f"| classical mean | {result['classical_mean_deg']} |",
        f"| classical p90 | {result['classical_p90_deg']} |",
        f"| median RANSAC inliers | {result['median_inliers']} |",
        "",
        "This is a calibrated classical baseline, not the FlyRot model. It uses RGB only and does not use IMU, depth, or segmentation.",
        "",
    ]
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
