import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from evaluate_directional_residual import _distribution, _macro_bootstrap, balanced_indices  # noqa: E402


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


def test_macro_bootstrap_is_group_level_and_deterministic():
    rows = [
        {"trajectory": "a", "geodesic_error_deg": 1.0},
        {"trajectory": "a", "geodesic_error_deg": 3.0},
        {"trajectory": "b", "geodesic_error_deg": 5.0},
    ]
    first = _macro_bootstrap(rows, "trajectory", seed=4, replicates=100)
    second = _macro_bootstrap(rows, "trajectory", seed=4, replicates=100)
    assert first == second
    assert first["groups"] == 2
    assert first["mean_geodesic_deg"] == 3.5
