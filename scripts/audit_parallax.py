#!/usr/bin/env python3
"""Measure rotational versus translational image flow using TartanAir-V2 depth."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import cv2
import numpy as np

from flyrot.geometry.pose_conventions import load_tartanair_poses, pose_rows_to_world_camera


FRAME_NUMBER = re.compile(r"^(\d+)")


def frame_id(path: Path) -> int:
    match = FRAME_NUMBER.match(path.name)
    if not match:
        raise ValueError(f"cannot parse frame id: {path}")
    return int(match.group(1))


def decode_depth(path: Path) -> np.ndarray:
    packed = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if packed is None or packed.ndim != 3 or packed.shape[2] != 4:
        raise ValueError(f"expected 4-channel packed depth PNG: {path}")
    return np.ascontiguousarray(packed).view("<f4").reshape(packed.shape[:2])


def raw_trajectory_root(raw_root: Path, record: dict) -> Path:
    direct = raw_root / record["environment"] / record["difficulty"] / record["trajectory"]
    nested = raw_root / record["environment"] / record["difficulty"] / record["environment"] / record["difficulty"] / record["trajectory"]
    for candidate in (direct, nested):
        if (candidate / "depth_lcam_front").is_dir():
            return candidate
    raise FileNotFoundError(f"depth trajectory not found for {record['environment']}/{record['trajectory']}")


def project(points: np.ndarray, focal: float, center: float) -> np.ndarray:
    z = points[:, 2]
    return np.stack((focal * points[:, 0] / z + center, focal * points[:, 1] / z + center), axis=1)


def audit_pair(depth_path: Path, poses: np.ndarray, start: int, gap: int, focal: float, center: float, stride: int, depth_max: float) -> dict:
    depth = decode_depth(depth_path)
    height, width = depth.shape
    ys = np.arange(0, height, stride)
    xs = np.arange(0, width, stride)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    depth_values = depth[yy, xx]
    valid = np.isfinite(depth_values) & (depth_values > 1e-4) & (depth_values < depth_max)
    if not np.any(valid):
        return {"valid_pixels": 0}
    x = (xx[valid].astype(np.float64) - center) / focal
    y = (yy[valid].astype(np.float64) - center) / focal
    points_t = depth_values[valid].astype(np.float64)[:, None] * np.stack((x, y, np.ones_like(x)), axis=1)
    matrices = pose_rows_to_world_camera(poses[[start, start + gap]])
    r_rel = matrices[0, :3, :3].T @ matrices[1, :3, :3]
    t_rel = matrices[0, :3, :3].T @ (matrices[1, :3, 3] - matrices[0, :3, 3])
    points_rot = points_t @ r_rel
    points_full = (points_t - t_rel) @ r_rel
    origin = np.stack((xx[valid], yy[valid]), axis=1).astype(np.float64)
    flow_rot = project(points_rot, focal, center) - origin
    flow_full = project(points_full, focal, center) - origin
    flow_translation = flow_full - flow_rot
    rot_norm = np.linalg.norm(flow_rot, axis=1)
    full_norm = np.linalg.norm(flow_full, axis=1)
    translation_norm = np.linalg.norm(flow_translation, axis=1)
    return {
        "valid_pixels": int(len(rot_norm)),
        "rotation_flow_px_median": float(np.median(rot_norm)),
        "full_flow_px_median": float(np.median(full_norm)),
        "translation_flow_px_median": float(np.median(translation_norm)),
        "translation_to_full_ratio_median": float(np.median(translation_norm / np.maximum(full_norm, 1e-6))),
        "translation_below_full_25pct": float(np.mean(translation_norm < 0.25 * np.maximum(full_norm, 1e-6))),
        "translation_below_full_50pct": float(np.mean(translation_norm < 0.50 * np.maximum(full_norm, 1e-6))),
    }


def summarize(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {key: None for key in ("p25", "median", "p75", "p90")}
    p25, median, p75, p90 = np.percentile(values, [25, 50, 75, 90])
    return {"p25": float(p25), "median": float(median), "p75": float(p75), "p90": float(p90)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, default=Path("data/tartanair-v2"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gap", type=int, default=4)
    parser.add_argument("--max-pairs-per-record", type=int, default=8)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--focal", type=float, default=320.0)
    parser.add_argument("--center", type=float, default=320.0)
    parser.add_argument("--depth-max", type=float, default=100.0)
    args = parser.parse_args()
    records = json.loads(args.index.read_text(encoding="utf-8"))
    pair_results = []
    unavailable = []
    for record in records:
        try:
            root = raw_trajectory_root(args.raw_root, record)
        except FileNotFoundError:
            unavailable.append({"environment": record["environment"], "trajectory": record["trajectory"]})
            continue
        depth_paths = {frame_id(path): path for path in (root / "depth_lcam_front").glob("*.png")}
        if not Path(record["pose_path"]).is_file():
            unavailable.append(
                {
                    "environment": record["environment"],
                    "trajectory": record["trajectory"],
                    "reason": "indexed pose path is unavailable",
                }
            )
            continue
        poses = load_tartanair_poses(record["pose_path"])
        starts = np.linspace(0, len(poses) - args.gap - 1, min(args.max_pairs_per_record, len(poses) - args.gap), dtype=int)
        for start in sorted(set(int(value) for value in starts)):
            path = depth_paths.get(start)
            if path is None or start + args.gap not in depth_paths:
                continue
            result = audit_pair(path, poses, start, args.gap, args.focal, args.center, args.stride, args.depth_max)
            result.update({"environment": record["environment"], "trajectory": record["trajectory"], "start": start, "gap": args.gap})
            pair_results.append(result)
    fields = (
        "rotation_flow_px_median",
        "full_flow_px_median",
        "translation_flow_px_median",
        "translation_to_full_ratio_median",
        "translation_below_full_25pct",
        "translation_below_full_50pct",
    )
    aggregate = {field: summarize([float(item[field]) for item in pair_results if field in item]) for field in fields}
    output = {
        "gap": args.gap,
        "focal": args.focal,
        "center": args.center,
        "depth_max": args.depth_max,
        "pixel_stride": args.stride,
        "pair_count": len(pair_results),
        "aggregate": aggregate,
        "unavailable_records": unavailable,
        "pairs": pair_results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"pair_count": len(pair_results), "aggregate": aggregate, "unavailable_count": len(unavailable)}, indent=2))


if __name__ == "__main__":
    main()
