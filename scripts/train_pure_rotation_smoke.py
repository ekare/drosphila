#!/usr/bin/env python3
"""Train FlyRot on depth-free, homography-generated pure-rotation RGB pairs."""

from __future__ import annotations

import argparse
import json
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from flyrot.data.tartanair import default_split
from flyrot.geometry.pose_conventions import load_tartanair_poses, relative_camera_rotations
from flyrot.losses import flyrot_loss
from flyrot.models.flyrot_v0 import FlyRotV0
from flyrot.geometry.so3 import log_so3


NED_TO_OPTICAL = np.asarray([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]], dtype=np.float64)


def frame_key(path: Path) -> int:
    return int(path.name.split("_", 1)[0])


def evenly_spaced_indices(length: int, limit: int) -> list[int]:
    if limit >= length:
        return list(range(length))
    if limit <= 1:
        return [0]
    return sorted({round(index * (length - 1) / (limit - 1)) for index in range(limit)})


class PureRotationDataset(Dataset):
    def __init__(
        self,
        records: list[dict],
        frame_gap: int = 4,
        image_size: int = 128,
        target_direction: str = "camera_relative",
        homography_direction: str = "image_motion",
    ) -> None:
        self.records = records
        self.frame_gap = frame_gap
        self.image_size = image_size
        self.target_direction = target_direction
        self.homography_direction = homography_direction
        self._loaded = []
        self._samples: list[tuple[int, int]] = []
        self._cache: dict[int, dict[str, torch.Tensor]] = {}
        for record_index, record in enumerate(records):
            paths = sorted(Path(record["rgb_root"]).glob("*.png"), key=frame_key)
            poses = load_tartanair_poses(record["pose_path"])
            if len(paths) != len(poses):
                raise ValueError(f"RGB/pose mismatch for {record['environment']}/{record['trajectory']}")
            self._loaded.append({"paths": paths, "poses": poses})
            self._samples.extend((record_index, start) for start in range(len(paths) - 2 * frame_gap))

    def __len__(self) -> int:
        return len(self._samples)

    def _read(self, path: Path) -> np.ndarray:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        return image.astype(np.float32) / 255.0

    def _make_sample(self, index: int) -> dict[str, torch.Tensor]:
        record_index, start = self._samples[index]
        loaded = self._loaded[record_index]
        poses = loaded["poses"]
        base = self._read(loaded["paths"][start])
        focal = self.image_size / 2.0
        center = (self.image_size - 1) / 2.0
        camera = np.asarray([[focal, 0.0, center], [0.0, focal, center], [0.0, 0.0, 1.0]], dtype=np.float64)
        camera_inverse = np.linalg.inv(camera)
        frames = [base]
        for offset in (self.frame_gap, 2 * self.frame_gap):
            rotation_ned = relative_camera_rotations(poses[[start, start + offset]])[0]
            if self.homography_direction == "image_motion":
                rotation_ned = rotation_ned.T
            rotation_optical_motion = NED_TO_OPTICAL @ rotation_ned @ NED_TO_OPTICAL.T
            homography = camera @ rotation_optical_motion @ camera_inverse
            warped = cv2.warpPerspective(
                base,
                homography,
                (self.image_size, self.image_size),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101,
            )
            frames.append(warped)
        rotation_ned = relative_camera_rotations(poses[[start, start + 2 * self.frame_gap]])[0]
        inverse = self.target_direction in {"image_motion", "optical_image_motion"}
        optical = self.target_direction in {"optical_relative", "optical_image_motion"}
        if inverse:
            rotation_ned = rotation_ned.T
        if optical:
            rotation_ned = NED_TO_OPTICAL @ rotation_ned @ NED_TO_OPTICAL.T
        target = log_so3(rotation_ned[None])[0]
        return {
            "frames": torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).contiguous(),
            "target_rotation_vector": torch.from_numpy(target.astype(np.float32)),
        }

    def preload(self, indices: list[int], workers: int = 8) -> None:
        unique = sorted(set(indices))
        with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
            samples = list(executor.map(self._make_sample, unique))
        self._cache.update(zip(unique, samples))

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if index not in self._cache:
            self._cache[index] = self._make_sample(index)
        return self._cache[index]


def evaluate(model: torch.nn.Module, datasets: list[PureRotationDataset], device: torch.device, limit: int) -> dict:
    model.eval()
    model_errors: list[float] = []
    zero_errors: list[float] = []
    with torch.no_grad():
        for dataset in datasets:
            loader = DataLoader(Subset(dataset, evenly_spaced_indices(len(dataset), min(limit, len(dataset)))), batch_size=32)
            for batch in loader:
                frames = batch["frames"].to(device)
                target = batch["target_rotation_vector"].to(device)
                prediction = model(frames)["rotation_vector"][:, -1]
                model_errors.extend((prediction - target).square().mean(dim=1).cpu().tolist())
                zero_errors.extend(target.square().mean(dim=1).cpu().tolist())
    model_mse = float(np.mean(model_errors)) if model_errors else None
    zero_mse = float(np.mean(zero_errors)) if zero_errors else None
    return {"model_mse": model_mse, "zero_mse": zero_mse, "model_below_zero": bool(model_mse is not None and model_mse < zero_mse)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--max-train-samples", type=int, default=1024)
    parser.add_argument("--max-val-samples", type=int, default=512)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--frame-gap", type=int, default=4)
    parser.add_argument(
        "--target-direction",
        choices=("camera_relative", "image_motion", "optical_relative", "optical_image_motion"),
        default="optical_image_motion",
    )
    parser.add_argument("--direction-scales", default="1,2")
    parser.add_argument("--homography-direction", choices=("camera_relative", "image_motion"), default="image_motion")
    parser.add_argument("--appearance-normalized", action="store_true")
    parser.add_argument("--scale-separated", action="store_true")
    parser.add_argument("--magnitude-aware", action="store_true")
    parser.add_argument("--least-squares-basis", action="store_true")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default=None, help="torch device; defaults to CUDA when available, otherwise CPU")
    parser.add_argument("--preload-workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    direction_scales = tuple(int(value) for value in args.direction_scales.split(",") if value)
    if not direction_scales or any(value < 1 for value in direction_scales):
        raise ValueError("direction-scales must contain positive integers")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    records = json.loads(args.index.read_text(encoding="utf-8"))
    splits = default_split(records)
    train_dataset = PureRotationDataset(
        splits["train"],
        frame_gap=args.frame_gap,
        image_size=args.image_size,
        target_direction=args.target_direction,
        homography_direction=args.homography_direction,
    )
    validation_datasets = [
        PureRotationDataset(
            [record],
            frame_gap=args.frame_gap,
            image_size=args.image_size,
            target_direction=args.target_direction,
            homography_direction=args.homography_direction,
        )
        for record in splits["validation"]
    ]
    train_indices = evenly_spaced_indices(len(train_dataset), min(args.max_train_samples, len(train_dataset)))
    train_dataset.preload(train_indices, workers=args.preload_workers)
    for dataset in validation_datasets:
        dataset.preload(evenly_spaced_indices(len(dataset), min(args.max_val_samples, len(dataset))), workers=args.preload_workers)
    loader = DataLoader(Subset(train_dataset, train_indices), batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    requested_device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested_device)
    model = FlyRotV0(
        scales=direction_scales,
        appearance_normalized=args.appearance_normalized,
        scale_separated=args.scale_separated,
        magnitude_aware=args.magnitude_aware,
        least_squares_basis=args.least_squares_basis,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    event_path = args.run_dir / "events.jsonl"
    best_val = float("inf")
    iterator = iter(loader)
    for step in range(args.steps):
        model.train()
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        frames = batch["frames"].to(device)
        target = batch["target_rotation_vector"].to(device)
        output = model(frames)
        prediction = output["rotation_vector"][:, -1]
        components = flyrot_loss(prediction, target, output["log_variance"][:, -1])
        optimizer.zero_grad(set_to_none=True)
        components["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        completed = step + 1
        if completed % args.eval_interval == 0 or completed == args.steps:
            validation = evaluate(model, validation_datasets, device, args.max_val_samples)
            state = {"model": model.state_dict(), "step": completed, "config": vars(args)}
            torch.save(state, args.run_dir / "latest.pt")
            if validation["model_mse"] < best_val:
                best_val = validation["model_mse"]
                torch.save(state, args.run_dir / "best.pt")
            event = {"step": completed, "validation": validation, "train_loss": float(components["total"].detach().cpu())}
            with event_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event) + "\n")
            print(json.dumps(event), flush=True)


if __name__ == "__main__":
    main()
