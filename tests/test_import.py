from importlib.metadata import version

from flyrot import __version__
from flyrot.data.discovery import discover_tartanair
from flyrot.data.tartanair import TartanAirWindowDataset, default_split


def test_package_imports():
    assert __version__ == "0.3.1"
    assert __version__ == version("flyrot")
    assert callable(discover_tartanair)
    assert TartanAirWindowDataset is not None
    assert callable(default_split)
