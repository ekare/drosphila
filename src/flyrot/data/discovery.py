"""Read-only discovery of the local TartanAir directory layout."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


@dataclass(frozen=True)
class TrajectoryRecord:
    dataset_root: str
    environment: str
    difficulty: str
    trajectory: str
    camera: str
    trajectory_root: str
    rgb_root: str
    pose_path: str | None
    rgb_count: int
    pose_count: int
    depth_count: int
    flow_count: int
    segmentation_count: int
    imu_present: bool
    resolution: list[int] | None
    pose_format: str | None
    notes: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


def _count_images(path: Path) -> int:
    if not path.is_dir():
        return 0
    return sum(1 for item in path.iterdir() if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES)


def _first_resolution(path: Path) -> list[int] | None:
    if not path.is_dir():
        return None
    for item in sorted(path.iterdir()):
        if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES:
            with Image.open(item) as image:
                return [int(image.width), int(image.height)]
    return None


def _find_asset_dir(trajectory_root: Path, prefix: str, camera: str) -> Path | None:
    candidates = [
        trajectory_root / f"{prefix}_{camera}",
        trajectory_root / f"{prefix}{camera}",
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    for candidate in trajectory_root.iterdir():
        if candidate.is_dir() and candidate.name.lower().startswith(prefix.lower()):
            if camera.lower() in candidate.name.lower():
                return candidate
    return None


def discover_tartanair(root: str | Path, environments: set[str] | None = None) -> list[TrajectoryRecord]:
    """Discover complete-looking trajectories without modifying the dataset."""

    dataset_root = Path(root).expanduser().resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(dataset_root)
    records: list[TrajectoryRecord] = []
    for environment_root in sorted(item for item in dataset_root.iterdir() if item.is_dir()):
        if environment_root.name.startswith(".") or ".extracting" in environment_root.name:
            continue
        if environments and environment_root.name not in environments:
            continue
        for difficulty_root in sorted(item for item in environment_root.iterdir() if item.is_dir() and item.name.startswith("Data_")):
            candidates = {
                item
                for item in difficulty_root.rglob("P[0-9][0-9][0-9]")
                if item.is_dir() and ".extracting" not in str(item)
            }
            for trajectory_root in sorted(candidates):
                image_dirs = sorted(
                    item for item in trajectory_root.rglob("image_*")
                    if item.is_dir() and ".extracting" not in str(item)
                )
                for rgb_root in image_dirs:
                    camera = rgb_root.name.removeprefix("image_")
                    pose_candidates = sorted(trajectory_root.rglob(f"pose_{camera}.txt"))
                    pose_path = pose_candidates[0] if pose_candidates else None
                    pose_count = sum(1 for _ in pose_path.open(encoding="utf-8")) if pose_path else 0
                    depth_root = _find_asset_dir(trajectory_root, "depth", camera)
                    flow_root = _find_asset_dir(trajectory_root, "flow", camera)
                    segmentation_root = _find_asset_dir(trajectory_root, "seg", camera)
                    notes: list[str] = []
                    if pose_count and pose_count != _count_images(rgb_root):
                        notes.append("rgb_pose_count_mismatch")
                    if ".extracting" in str(trajectory_root):
                        notes.append("incomplete_extraction_path")
                    records.append(
                        TrajectoryRecord(
                            dataset_root=str(dataset_root),
                            environment=environment_root.name,
                            difficulty=difficulty_root.name,
                            trajectory=trajectory_root.name,
                            camera=camera,
                            trajectory_root=str(trajectory_root),
                            rgb_root=str(rgb_root),
                            pose_path=str(pose_path) if pose_path else None,
                            rgb_count=_count_images(rgb_root),
                            pose_count=pose_count,
                            depth_count=_count_images(depth_root) if depth_root else 0,
                            flow_count=_count_images(flow_root) if flow_root else 0,
                            segmentation_count=_count_images(segmentation_root) if segmentation_root else 0,
                            imu_present=(trajectory_root / "imu").is_dir(),
                            resolution=_first_resolution(rgb_root),
                            pose_format="tx ty tz qx qy qz qw" if pose_path else None,
                            notes=notes,
                        )
                    )
    return records
