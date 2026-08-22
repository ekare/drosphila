import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from evaluate_directional_residual import _distribution, balanced_indices  # noqa: E402


def test_balanced_indices_is_deterministic_and_covers_records():
    selected = balanced_indices([10, 4, 2], 8)
    assert selected == balanced_indices([10, 4, 2], 8)
    assert [len(group) for group in selected] == [3, 3, 2]
    assert all(index in range(length) for length, group in zip([10, 4, 2], selected) for index in group)


def test_balanced_indices_caps_at_available_windows():
    selected = balanced_indices([2, 1], 99)
    assert selected == [[0, 1], [0]]


def test_distribution_preserves_signed_delta_histogram():
    distribution = _distribution([-0.4, -0.1, 0.0, 0.2])
    edges = distribution["histogram"]["edges"]
    assert edges[0] < 0.0
    assert sum(distribution["histogram"]["counts"]) == 4
