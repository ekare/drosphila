import os
from pathlib import Path

import pytest

from flyrot.data.discovery import discover_tartanair


def test_discovery_finds_verified_house_sequence():
    root = Path(os.environ.get("FLYROT_DATA_ROOT", "data/tartanair-v2"))
    if not root.is_dir():
        pytest.skip("set FLYROT_DATA_ROOT to run the dataset discovery integration test")
    records = discover_tartanair(root, {"House"})
    house_p000 = [record for record in records if record.trajectory == "P000" and record.camera == "lcam_front"]
    assert house_p000
    assert house_p000[0].rgb_count == 680
    assert house_p000[0].pose_count == 680
    assert house_p000[0].resolution == [640, 640]
