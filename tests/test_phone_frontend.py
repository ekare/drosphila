import torch

from flyrot.models.phone_frontend import PhonePolarityFrontend


def test_phone_frontend_candidates_are_fixed_and_shape_stable():
    frames = torch.rand(1, 5, 1, 32, 48)
    for variant in PhonePolarityFrontend.VARIANTS:
        frontend = PhonePolarityFrontend(variant)
        result = frontend(frames)
        assert set(result) == {"pre", "adapted", "on", "off"}
        assert result["on"].shape == frames.shape
        assert torch.all(result["on"] >= 0)
        assert torch.all(result["off"] >= 0)
        assert len(list(frontend.parameters())) == 0
