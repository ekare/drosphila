import torch

from flyrot.real_texture import decode_direction_population
from flyrot.validation_contract import (
    CANONICAL_DIRECTION_NAMES,
    d4_audit,
    direction_index,
    direction_vectors,
    photoreceptor_variants,
    pure_edge_stimulus,
    reorder_direction_vector,
    response_metrics,
    transform_response,
    d4_matrices,
)


def test_canonical_image_plane_order_and_vectors():
    assert CANONICAL_DIRECTION_NAMES == ("E", "SE", "S", "SW", "W", "NW", "N", "NE")
    vectors = direction_vectors()
    assert torch.allclose(vectors[0], torch.tensor([1.0, 0.0]))
    assert vectors[1, 1] > 0
    assert vectors[2, 0].abs() < 1e-6 and vectors[2, 1] > 0
    assert vectors[6, 1] < 0
    assert direction_index(0) == 0
    assert direction_index(90) == 2
    assert direction_index(270) == 6


def test_legacy_order_conversion_is_explicit_and_invertible():
    legacy = torch.arange(8, dtype=torch.float32)
    canonical = reorder_direction_vector(legacy)
    assert canonical.tolist() == [0, 7, 6, 5, 4, 3, 2, 1]
    assert torch.equal(reorder_direction_vector(canonical), legacy)


def test_population_tie_and_semantic_metrics():
    response = torch.eye(8)
    metrics = response_metrics(response, torch.arange(8))
    assert metrics["argmax_accuracy"] == 1.0
    assert metrics["tie_aware_accuracy"] == 1.0
    assert metrics["population_vector_median_error_deg"] == 0.0
    tie = torch.ones(8, 8)
    tie_metrics = response_metrics(tie, torch.arange(8))
    assert tie_metrics["tie_aware_accuracy"] == 1.0
    assert tie_metrics["adjacent_tie_fraction"] == 1.0


def test_d4_audit_detects_y_flip_without_weight_change():
    expected = torch.arange(8)
    canonical = torch.eye(8)
    observed = transform_response(canonical, d4_matrices()["flip_y"])
    audit = d4_audit(observed, expected)
    assert audit["best_physical_mapping"] == "flip_y"
    assert audit["best_population_median_error_deg"] == 0.0


def test_pure_edge_has_only_one_polarity_transition_per_pixel():
    for kind, sign in (("on", 1), ("off", -1)):
        frames, valid = pure_edge_stimulus(kind, angle_deg=45, displacement=4)
        delta = frames[1:, 0] - frames[:-1, 0]
        changes = delta[:, valid[0].bool()]
        assert torch.all(changes * sign >= -1e-7)
        positive = (changes > 1e-7).any(dim=0)
        negative = (changes < -1e-7).any(dim=0)
        assert not torch.any(positive & negative)
        traces = photoreceptor_variants(frames)
        assert set(traces) == {"P0", "P1", "P2"}
        assert all(value["on"].shape == (1, 5, 1, 128, 160) for value in traces.values())


def test_real_texture_decoder_canonical_conversion_keeps_shape():
    energy = torch.zeros(1, 2, 2 * 2 * 8, 8, 8)
    energy[:, :, 0] = 1.0
    legacy = decode_direction_population(energy, (1, 2), direction_order="legacy")
    canonical = decode_direction_population(energy, (1, 2), direction_order="canonical")
    assert legacy["velocity"].shape == canonical["velocity"].shape
    assert torch.isfinite(canonical["velocity"]).all()
    assert canonical["direction_order"] == "canonical"
