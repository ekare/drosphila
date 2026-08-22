#!/usr/bin/env python3
"""Audit pose/image rotation direction over many cached TartanAir trajectories."""

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


def compare_pair(image_a: Path, image_b: Path, poses: np.ndarray, start: int, gap: int, focal: float) -> dict:
    first = cv2.imread(str(image_a), cv2.IMREAD_GRAYSCALE)
    second = cv2.imread(str(image_b), cv2.IMREAD_GRAYSCALE)
    if first is None or second is None:
        return {"tracked": 0, "winner": "unreadable"}
    points = cv2.goodFeaturesToTrack(first, maxCorners=500, qualityLevel=0.01, minDistance=5)
    if points is None:
        return {"tracked": 0, "winner": "no_features"}
    next_points, status, _ = cv2.calcOpticalFlowPyrLK(first, second, points, None)
    status = status.reshape(-1).astype(bool)
    old = points.reshape(-1, 2)[status]
    new = next_points.reshape(-1, 2)[status]
    observed = new - old
    center = np.array([first.shape[1] / 2.0, first.shape[0] / 2.0])
    relative = relative_camera_rotations(poses[[start, start + gap]])[0]
    candidates = {"r_rel_transpose": rotational_flow(relative.T, old, focal, center), "r_rel": rotational_flow(relative, old, focal, center)}
    scores = {}
    observed_norm = np.linalg.norm(observed, axis=1)
    valid_observed = observed_norm > 0.5
    for name, predicted in candidates.items():
        predicted_norm = np.linalg.norm(predicted, axis=1)
        valid = valid_observed & (predicted_norm > 1e-4) & np.isfinite(predicted).all(axis=1)
        if not np.any(valid):
            scores[name] = None
            continue
        scores[name] = float(np.median(np.sum(observed[valid] * predicted[valid], axis=1) / (observed_norm[valid] * predicted_norm[valid])))
    transpose = scores["r_rel_transpose"]
    inverse = scores["r_rel"]
    winner = "tie"
    if transpose is not None and inverse is not None:
        winner = "r_rel_transpose" if transpose > inverse else "r_rel" if inverse > transpose else "tie"
    return {"tracked": int(len(old)), "scores": scores, "winner": winner}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    parser.add_argument("--gap", type=int, default=4)
    parser.add_argument("--pairs-per-record", type=int, default=8)
    parser.add_argument("--focal", type=float, default=320.0)
    args = parser.parse_args()

    records = json.loads(args.index.read_text(encoding="utf-8"))
    records = [r for r in records if r["environment"] in {"House", "Office", "ModernCityDowntown", "Cyberpunk"}]
    results = []
    for record in records:
        rgb_root = Path(record["rgb_root"])
        images = sorted(rgb_root.glob("*.png"))
        poses = load_tartanair_poses(record["pose_path"])
        if len(images) != len(poses):
            raise RuntimeError(f"RGB/pose mismatch: {rgb_root}")
        max_start = len(images) - args.gap - 1
        starts = sorted({round(i * max_start / max(1, args.pairs_per_record - 1)) for i in range(args.pairs_per_record)})
        for start in starts:
            result = compare_pair(images[start], images[start + args.gap], poses, start, args.gap, args.focal)
            result.update({"environment": record["environment"], "trajectory": record["trajectory"], "start": start, "gap": args.gap})
            results.append(result)

    valid = [r for r in results if r["winner"] in {"r_rel_transpose", "r_rel", "tie"}]
    wins = {name: sum(r["winner"] == name for r in valid) for name in ("r_rel_transpose", "r_rel", "tie")}
    transpose_scores = [r["scores"]["r_rel_transpose"] for r in valid if r["scores"].get("r_rel_transpose") is not None]
    inverse_scores = [r["scores"]["r_rel"] for r in valid if r["scores"].get("r_rel") is not None]
    summary = {
        "records": len(records),
        "pairs_attempted": len(results),
        "pairs_with_scores": len(transpose_scores),
        "gap": args.gap,
        "focal": args.focal,
        "wins": wins,
        "median_cosine": {
            "r_rel_transpose": float(np.median(transpose_scores)) if transpose_scores else None,
            "r_rel": float(np.median(inverse_scores)) if inverse_scores else None,
        },
        "interpretation": "directional sanity check only; translation/parallax and focal assumptions remain",
        "pairs": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Multi-trajectory pose/image rotation audit",
        "",
        f"Records: `{len(records)}`; pairs: `{len(results)}`; gap: `{args.gap}`; focal: `{args.focal:.1f}` px",
        "",
        f"Wins: `R_rel.T={wins['r_rel_transpose']}`, `R_rel={wins['r_rel']}`, ties=`{wins['tie']}`",
        f"Median cosine: `R_rel.T={summary['median_cosine']['r_rel_transpose']}`, `R_rel={summary['median_cosine']['r_rel']}`",
        "",
        "This is a directional sanity check only; translation/parallax and focal assumptions are not removed.",
    ]
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "pairs"}, indent=2))


if __name__ == "__main__":
    main()
