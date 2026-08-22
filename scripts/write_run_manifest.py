#!/usr/bin/env python3
"""Write a portable provenance manifest for one completed or planned run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from flyrot.provenance import build_run_manifest, write_run_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--dataset-index", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--split")
    parser.add_argument("--command", nargs=argparse.REMAINDER, default=[])
    args = parser.parse_args()
    manifest = build_run_manifest(
        repo_root=args.repo_root,
        command=args.command,
        seed=args.seed,
        split=args.split,
        dataset_index=args.dataset_index,
        config_path=args.config,
        checkpoint_path=args.checkpoint,
    )
    write_run_manifest(args.output, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
