#!/usr/bin/env python3
"""Verify that the built wheel contains the complete importable package."""

from pathlib import Path
from zipfile import ZipFile


def main() -> None:
    wheels = sorted(Path("dist").glob("*.whl"))
    if not wheels:
        raise SystemExit("no wheel found in dist/")
    wheel = wheels[-1]
    with ZipFile(wheel) as archive:
        names = set(archive.namelist())
    required = {"flyrot/__init__.py", "flyrot/diagnostics.py", "flyrot/data/__init__.py"}
    missing = required - names
    if missing or not any(name.startswith("flyrot/data/") for name in names):
        raise SystemExit(f"wheel package check failed: missing={sorted(missing)}")
    print(wheel)


if __name__ == "__main__":
    main()
