#!/usr/bin/env python3
"""Build a compact JSON/Markdown index without copying dataset files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from flyrot.data.discovery import discover_tartanair


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    parser.add_argument("--environment", action="append", dest="environments")
    args = parser.parse_args()

    selected = set(args.environments) if args.environments else None
    records = [record.to_dict() for record in discover_tartanair(args.root, selected)]
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# TartanAir dataset index",
        "",
        f"Root: `{args.root}`",
        f"Trajectory-camera records: `{len(records)}`",
        "",
        "| environment | difficulty | trajectory | camera | rgb | pose | depth | flow | seg | resolution | notes |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for record in records:
        resolution = "x".join(map(str, record["resolution"])) if record["resolution"] else "-"
        notes = ", ".join(record["notes"]) or "-"
        lines.append(
            f"| {record['environment']} | {record['difficulty']} | {record['trajectory']} | {record['camera']} | "
            f"{record['rgb_count']} | {record['pose_count']} | {record['depth_count']} | {record['flow_count']} | "
            f"{record['segmentation_count']} | {resolution} | {notes} |"
        )
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"indexed {len(records)} trajectory-camera records")


if __name__ == "__main__":
    main()
