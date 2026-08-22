import torch

from flyrot.t4t5 import StimulusSpec, evaluate_bank, make_stimulus


def test_t4t5_stimulus_is_deterministic_and_non_square():
    spec = StimulusSpec("moving_bright_edge", angle_deg=0, displacement=4, width=64, height=48, seed=7)
    first, mask = make_stimulus(spec)
    second, second_mask = make_stimulus(spec)
    assert first.shape == (5, 1, 48, 64)
    assert torch.equal(first, second)
    assert torch.equal(mask, second_mask)


def test_t4t5_bank_reports_separate_polarities():
    frames, mask = make_stimulus(StimulusSpec("moving_bright_bar", angle_deg=0, displacement=4, seed=3))
    result = evaluate_bank(frames, mask, scales=(1, 2, 4))
    assert result["on_response"].shape == (8,)
    assert result["off_response"].shape == (8,)
    assert torch.isfinite(result["response"]).all()
