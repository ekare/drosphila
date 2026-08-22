"""FlyRot research package."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("flyrot")
except PackageNotFoundError:
    # Source-tree fallback for an uninstalled checkout. The authoritative
    # release version remains the project metadata in pyproject.toml.
    __version__ = "0.3.0"
