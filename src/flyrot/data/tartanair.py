"""TartanAir RGB+pose window dataset."""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from flyrot.geometry.pose_conventions import NED_TO_OPTICAL, load_tartanair_poses, relative_camera_rotations
from flyrot.geometry.so3 import log_so3


_FRAME_NUMBER = re.compile(r"^(\d+)")

# TartanAir's camera coordinates are NED: [forward, right, down].  The
# pinhole flow bases use [right, down, forward].
def _target_rotation_vectors(relative: np.ndarray, target_direction: str) -> np.ndarray:
    if target_direction not in {"camera_relative", "image_motion", "optical_relative", "optical_image_motion"}:
        raise ValueError(f"unknown target direction {target_direction!r}")
    if target_direction in {"image_motion", "optical_image_motion"}:
        relative = relative.transpose(0, 2, 1)
    if target_direction in {"optical_relative", "optical_image_motion"}:
        relative = NED_TO_OPTICAL @ relative @ NED_TO_OPTICAL.T
    return log_so3(relative)


def _frame_key(path: Path) -> tuple[int, str]:
    match = _FRAME_NUMBER.match(path.name)
    return (int(match.group(1)) if match else -1, path.name)


def default_split(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Split complete trajectory records without frame-level leakage."""

    train_environments = {"House", "Office"}
    test_environments = {"ModernCityDowntown", "Cyberpunk"}
    train_candidates = [r for r in records if r["environment"] in train_environments]
    test_records = [r for r in records if r["environment"] in test_environments]
    validation: list[dict[str, Any]] = []
    train: list[dict[str, Any]] = []
    for environment in sorted(train_environments):
        environment_records = sorted(
            [r for r in train_candidates if r["environment"] == environment],
            key=lambda r: r["trajectory"],
        )
        if not environment_records:
            continue
        validation.append(environment_records[-1])
        train.extend(environment_records[:-1])
    return {"train": train, "validation": validation, "test": sorted(test_records, key=lambda r: (r["environment"], r["trajectory"]))}


class TartanAirWindowDataset(Dataset):
    def __init__(
        self,
        records: list[dict[str, Any]],
        window_length: int = 3,
        frame_gap: int = 1,
        image_size: int | tuple[int, int] = 128,
        preload_images: bool = False,
        preload_workers: int = 1,
        target_direction: str = "camera_relative",
    ) -> None:
        if window_length < 2 or frame_gap < 1:
            raise ValueError("window_length must be >=2 and frame_gap must be >=1")
        if target_direction not in {
            "camera_relative",
            "image_motion",
            "optical_relative",
            "optical_image_motion",
        }:
            raise ValueError(f"unknown target_direction: {target_direction}")
        self.records = records
        self.window_length = window_length
        self.frame_gap = frame_gap
        self.target_direction = target_direction
        if isinstance(image_size, int):
            self.image_size = (image_size, image_size)
        else:
            self.image_size = (int(image_size[0]), int(image_size[1]))
        self.preload_images = preload_images
        self.preload_workers = max(1, preload_workers)
        self._loaded: list[dict[str, Any]] = []
        self._samples: list[tuple[int, int]] = []
        executor = ThreadPoolExecutor(max_workers=self.preload_workers) if self.preload_images else None
        for record_index, record in enumerate(records):
            rgb_root = Path(record["rgb_root"])
            frame_paths = sorted(
                [item for item in rgb_root.iterdir() if item.suffix.lower() in {".png", ".jpg", ".jpeg"}],
                key=_frame_key,
            )
            if len(frame_paths) != int(record["rgb_count"]):
                raise ValueError(f"indexed RGB count changed for {rgb_root}")
            pose_path = record.get("pose_path")
            if not pose_path:
                raise ValueError(f"pose is required for RGB+pose dataset: {rgb_root}")
            poses = load_tartanair_poses(pose_path)
            if len(frame_paths) != len(poses):
                raise ValueError(f"RGB/pose mismatch for {rgb_root}: {len(frame_paths)} vs {len(poses)}")
            frame_numbers = [_frame_key(path)[0] for path in frame_paths]
            if any(next_id != current_id + 1 for current_id, next_id in zip(frame_numbers, frame_numbers[1:])):
                raise ValueError(f"non-contiguous frame ids in {rgb_root}")
            loaded = {"record": record, "frame_paths": frame_paths, "poses": poses}
            if self.preload_images:
                loaded["images"] = torch.stack(list(executor.map(self._read_image, frame_paths)))
            self._loaded.append(loaded)
            span = (window_length - 1) * frame_gap
            self._samples.extend((record_index, start) for start in range(0, len(frame_paths) - span))
        if executor is not None:
            executor.shutdown(wait=True)

    def __len__(self) -> int:
        return len(self._samples)

    def _read_image(self, path: Path) -> torch.Tensor:
        with Image.open(path) as image:
            image = image.convert("RGB")
            target_width, target_height = self.image_size
            if target_width == target_height:
                # Preserve the camera aspect ratio before making a square
                # model input. TartanAir-V2 is already square, so this is a
                # no-op for the active dataset; it avoids horizontal stretch
                # for future 4:3 or other non-square sources.
                source_width, source_height = image.size
                scale = max(target_width / source_width, target_height / source_height)
                resized_size = (
                    max(target_width, round(source_width * scale)),
                    max(target_height, round(source_height * scale)),
                )
                image = image.resize(resized_size, Image.Resampling.BILINEAR)
                left = (resized_size[0] - target_width) // 2
                top = (resized_size[1] - target_height) // 2
                image = image.crop((left, top, left + target_width, top + target_height))
            else:
                image = image.resize(self.image_size, Image.Resampling.BILINEAR)
            array = np.asarray(image, dtype=np.float32) / 255.0
        return torch.from_numpy(array).permute(2, 0, 1).contiguous()

    def __getitem__(self, index: int) -> dict[str, Any]:
        record_index, start = self._samples[index]
        loaded = self._loaded[record_index]
        frame_indices = [start + offset * self.frame_gap for offset in range(self.window_length)]
        if self.preload_images:
            images = loaded["images"][frame_indices]
        else:
            images = torch.stack([self._read_image(loaded["frame_paths"][frame_index]) for frame_index in frame_indices])
        pair_poses = loaded["poses"][frame_indices]
        pair_relative = relative_camera_rotations(pair_poses)
        pair_targets = _target_rotation_vectors(pair_relative, self.target_direction)
        endpoint_relative = relative_camera_rotations(pair_poses[[0, -1]])
        target = torch.from_numpy(_target_rotation_vectors(endpoint_relative, self.target_direction)[0]).to(torch.float32)
        record = loaded["record"]
        return {
            "frames": images,
            "target_rotation_vector": target,
            "target_rotation_vectors": torch.from_numpy(pair_targets).to(torch.float32),
            "metadata": {
                "environment": record["environment"],
                "difficulty": record["difficulty"],
                "trajectory": record["trajectory"],
                "camera": record["camera"],
                "start_frame": int(frame_indices[0]),
                "end_frame": int(frame_indices[-1]),
                "frame_gap": self.frame_gap,
                "target_direction": self.target_direction,
            },
        }
