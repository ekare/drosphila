"""Local delayed correlation detectors inspired by direction-selective cells."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


DIRECTION_NAMES = ("E", "NE", "N", "NW", "W", "SW", "S", "SE")
DIRECTION_OFFSETS = ((1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1))


def shift_zero(values: torch.Tensor, dx: int, dy: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Shift H/W without wrap-around and return a valid-pixel mask."""

    if values.ndim != 4:
        raise ValueError(f"expected N,C,H,W, got {tuple(values.shape)}")
    _, _, height, width = values.shape
    padding = (max(dx, 0), max(-dx, 0), max(dy, 0), max(-dy, 0))
    padded = F.pad(values, padding)
    x0 = max(-dx, 0)
    y0 = max(-dy, 0)
    shifted = padded[:, :, y0 : y0 + height, x0 : x0 + width]
    mask = values.new_ones((values.shape[0], 1, height, width))
    if dx > 0:
        mask[..., :, :dx] = 0
    elif dx < 0:
        mask[..., :, dx:] = 0
    if dy > 0:
        mask[..., :dy, :] = 0
    elif dy < 0:
        mask[..., dy:, :] = 0
    return shifted, mask


class DirectionCellBank(nn.Module):
    def __init__(self, scales: tuple[int, ...] = (1, 2)) -> None:
        super().__init__()
        self.scales = tuple(scales)
        self.direction_gain = nn.Parameter(torch.ones(2, len(DIRECTION_OFFSETS)))

    @property
    def channels(self) -> int:
        return 2 * len(self.scales) * len(DIRECTION_OFFSETS)

    def forward(
        self,
        on: torch.Tensor,
        off: torch.Tensor,
        return_components: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        if on.shape != off.shape or on.ndim != 5:
            raise ValueError("ON/OFF inputs must both have shape B,T,1,H,W")
        b, t, _, h, w = on.shape
        outputs = []
        masks = []
        on_outputs = []
        off_outputs = []
        for scale in self.scales:
            for direction, (dx0, dy0) in enumerate(DIRECTION_OFFSETS):
                dx, dy = dx0 * scale, dy0 * scale
                current_on = on[:, 1:].reshape(b * (t - 1), 1, h, w)
                delayed_on = on[:, :-1].reshape(b * (t - 1), 1, h, w)
                current_off = off[:, 1:].reshape(b * (t - 1), 1, h, w)
                delayed_off = off[:, :-1].reshape(b * (t - 1), 1, h, w)
                shifted_delayed_on, mask = shift_zero(delayed_on, dx, dy)
                shifted_current_on, _ = shift_zero(current_on, dx, dy)
                shifted_delayed_off, _ = shift_zero(delayed_off, dx, dy)
                shifted_current_off, _ = shift_zero(current_off, dx, dy)
                corr_on = shifted_delayed_on * current_on - delayed_on * shifted_current_on
                corr_off = shifted_delayed_off * current_off - delayed_off * shifted_current_off
                on_response = corr_on.clamp_min(0) * self.direction_gain[0, direction]
                off_response = corr_off.clamp_min(0) * self.direction_gain[1, direction]
                response = on_response + off_response
                outputs.append(response.reshape(b, t - 1, 1, h, w))
                masks.append(mask.reshape(b, t - 1, 1, h, w))
                if return_components:
                    on_outputs.append(on_response.clamp_min(0).reshape(b, t - 1, 1, h, w))
                    off_outputs.append(off_response.clamp_min(0).reshape(b, t - 1, 1, h, w))
        energy = torch.cat(outputs, dim=2)
        valid = torch.cat(masks, dim=2)
        if not return_components:
            return energy, valid
        return energy, valid, torch.cat(on_outputs, dim=2), torch.cat(off_outputs, dim=2)
