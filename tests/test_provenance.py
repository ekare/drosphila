import json

from flyrot.provenance import (
    build_run_manifest,
    normalize_dataset_manifest,
    sha256_file,
    sha256_json,
)


def test_dataset_manifest_is_path_free_and_order_stable():
    records = [
        {
            "dataset_root": "/private/source",
            "trajectory_root": "/private/source/House/Data_easy/P001",
            "rgb_root": "/private/source/image_lcam_front",
            "pose_path": "/private/source/pose_lcam_front.txt",
            "environment": "House",
            "difficulty": "Data_easy",
            "trajectory": "P001",
            "camera": "lcam_front",
            "rgb_count": 10,
            "pose_count": 10,
            "notes": [],
        }
    ]
    manifest = normalize_dataset_manifest(records)
    serialized = json.dumps(manifest)
    assert "/private" not in serialized
    assert manifest["records"][0]["rgb_count"] == 10
    assert sha256_json(manifest) == sha256_json(normalize_dataset_manifest(reversed(records)))


def test_run_manifest_hashes_artifacts_without_absolute_paths(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    config = repo / "config.yaml"
    checkpoint = repo / "checkpoint.pt"
    config.write_text("model: test\n", encoding="utf-8")
    checkpoint.write_bytes(b"checkpoint")
    manifest = build_run_manifest(
        repo_root=repo,
        command=["python", "train.py", "--seed", "7"],
        seed=7,
        split="validation",
        dataset_manifest={"schema": "test", "records": []},
        config_path=config,
        checkpoint_path=checkpoint,
    )
    assert manifest["config"]["sha256"] == sha256_file(config)
    assert manifest["checkpoint"]["sha256"] == sha256_file(checkpoint)
    assert manifest["config"]["path"] == "config.yaml"
    assert str(tmp_path) not in json.dumps(manifest)
