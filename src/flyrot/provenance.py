"""Portable, machine-auditable experiment provenance helpers.

Generated manifests record logical dataset identity and artifact hashes, not
local filesystem paths, hostnames, usernames, GPU UUIDs, or other private
execution context.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping


_PRIVATE_PATH_KEYS = {
    "dataset_root",
    "trajectory_root",
    "rgb_root",
    "depth_root",
    "flow_root",
    "segmentation_root",
    "pose_path",
    "rgb_path",
    "depth_path",
    "flow_path",
    "segmentation_path",
}


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest of a file without loading it into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    """Hash a JSON-compatible value using a deterministic representation."""

    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _portable_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Remove path-bearing fields while retaining dataset identity/counts."""

    result: dict[str, Any] = {}
    for key, value in record.items():
        if key in _PRIVATE_PATH_KEYS:
            continue
        if (key.endswith("_root") or key.endswith("_path")) and isinstance(value, str):
            continue
        result[str(key)] = value
    return result


def normalize_dataset_manifest(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Build a path-free, order-stable manifest from index records."""

    portable = [_portable_record(record) for record in records]
    portable.sort(
        key=lambda record: tuple(
            str(record.get(field, ""))
            for field in ("environment", "difficulty", "trajectory", "camera")
        )
    )
    return {"schema": "flyrot.dataset-manifest.v1", "records": portable}


def load_dataset_manifest(index_path: str | Path) -> dict[str, Any]:
    with Path(index_path).open(encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, list):
        raise ValueError("dataset index must contain a list of records")
    return normalize_dataset_manifest(records)


def _git_value(repo_root: Path, *args: str) -> str | None:
    completed = subprocess.run(
        ["git", *args], cwd=repo_root, check=False, capture_output=True, text=True
    )
    if completed.returncode:
        return None
    return completed.stdout.strip()


def git_snapshot(repo_root: str | Path = ".") -> dict[str, Any]:
    """Return commit/branch/dirty state without machine-specific metadata."""

    root = Path(repo_root).resolve()
    unstaged = subprocess.run(["git", "diff", "--quiet"], cwd=root, check=False).returncode != 0
    staged = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=root, check=False).returncode != 0
    return {
        "commit": _git_value(root, "rev-parse", "HEAD"),
        "branch": _git_value(root, "branch", "--show-current"),
        "describe": _git_value(root, "describe", "--always", "--tags"),
        "dirty": unstaged or staged,
    }


def environment_snapshot() -> dict[str, Any]:
    """Capture software/GPU facts without private IDs."""

    snapshot: dict[str, Any] = {"python": sys.version.split()[0]}
    try:
        import torch

        snapshot.update(
            {
                "torch": torch.__version__,
                "cuda_runtime": torch.version.cuda,
                "cuda_available": bool(torch.cuda.is_available()),
                "gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            }
        )
    except ImportError:
        snapshot.update({"torch": None, "cuda_runtime": None, "cuda_available": False, "gpu_names": []})
    return snapshot


def _relative_or_name(path: str | Path, repo_root: Path) -> str:
    candidate = Path(path)
    try:
        return candidate.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return candidate.name


def build_run_manifest(
    *,
    repo_root: str | Path = ".",
    command: Iterable[str],
    seed: int | None,
    split: str | None,
    dataset_manifest: Mapping[str, Any] | None = None,
    dataset_index: str | Path | None = None,
    config_path: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    outputs: Iterable[str | Path] = (),
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a path-free run manifest suitable for tracking with a report."""

    root = Path(repo_root).resolve()
    if dataset_manifest is None and dataset_index is not None:
        dataset_manifest = load_dataset_manifest(dataset_index)
    manifest: dict[str, Any] = {
        "schema": "flyrot.run-manifest.v1",
        "git": git_snapshot(root),
        "environment": environment_snapshot(),
        "command": [str(item) for item in command],
        "seed": seed,
        "split": split,
    }
    if dataset_manifest is not None:
        manifest["dataset"] = {
            "manifest_schema": dataset_manifest.get("schema"),
            "manifest_sha256": sha256_json(dataset_manifest),
            "record_count": len(dataset_manifest.get("records", [])),
        }
    for label, path in (("config", config_path), ("checkpoint", checkpoint_path)):
        if path is not None:
            manifest[label] = {"path": _relative_or_name(path, root), "sha256": sha256_file(path)}
    manifest["outputs"] = [_relative_or_name(path, root) for path in outputs]
    if extra:
        manifest["extra"] = dict(extra)
    return manifest


def write_run_manifest(path: str | Path, manifest: Mapping[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
