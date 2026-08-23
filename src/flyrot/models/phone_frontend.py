"""Small causal polarity frontends for the phone-vision gate."""

from __future__ import annotations

import torch
from torch import nn

from flyrot.validation_contract import photoreceptor_variants


class PhonePolarityFrontend(nn.Module):
    """Select one of the bounded P0-P3 sign-preserving frontend candidates.

    These candidates are fixed preprocessing variants. They contain no
    learned weights; selection must be made on the validation gate before any
    downstream model training.
    """

    VARIANTS = ("P0", "P1", "P2", "P3")

    def __init__(self, variant: str = "P2", local_window: int = 5) -> None:
        super().__init__()
        if variant not in self.VARIANTS:
            raise ValueError(f"variant must be one of {self.VARIANTS}")
        self.variant = variant
        self.local_window = local_window

    def forward(self, frames: torch.Tensor) -> dict[str, torch.Tensor]:
        values = photoreceptor_variants(frames, local_window=self.local_window)[self.variant if self.variant != "P3" else "P2"]
        if self.variant != "P3":
            return values
        adapted = values["adapted"]
        common_mode = adapted.mean(dim=(-1, -2), keepdim=True)
        residual = adapted - common_mode
        return {
            "pre": values["pre"],
            "adapted": residual,
            "on": residual.clamp_min(0),
            "off": (-residual).clamp_min(0),
        }


__all__ = ["PhonePolarityFrontend"]
