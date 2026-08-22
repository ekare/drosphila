"""Parameter-light temporal photoreceptor and ON/OFF split."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class Photoreceptor(nn.Module):
    def __init__(self, local_window: int = 5) -> None:
        super().__init__()
        if local_window % 2 == 0:
            raise ValueError("local_window must be odd")
        self.local_window = local_window
        self.log_adaptation = nn.Parameter(torch.tensor(0.0))

    def forward(self, frames: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ON/OFF temporal differences with shape ``B,T,1,H,W``.

        The first time step is a zero high-pass state so the downstream
        direction-cell stage produces one output for every input frame pair.
        """

        if frames.ndim != 5:
            raise ValueError(f"expected B,T,C,H,W input, got {tuple(frames.shape)}")
        if frames.shape[2] == 1:
            gray = frames
        elif frames.shape[2] == 3:
            gray = 0.299 * frames[:, :, 0:1] + 0.587 * frames[:, :, 1:2] + 0.114 * frames[:, :, 2:3]
        else:
            raise ValueError("frames must have one or three channels")
        gray = gray.clamp_min(0).add(1e-4).log()
        b, t, c, h, w = gray.shape
        flattened = gray.reshape(b * t, c, h, w)
        local_mean = F.avg_pool2d(flattened, self.local_window, stride=1, padding=self.local_window // 2)
        local_second = F.avg_pool2d(flattened.square(), self.local_window, stride=1, padding=self.local_window // 2)
        local_std = (local_second - local_mean.square()).clamp_min(1e-6).sqrt()
        adapted = ((flattened - local_mean) / local_std) * self.log_adaptation.exp().clamp(0.25, 4.0)
        adapted = adapted.reshape(b, t, c, h, w)
        difference = adapted[:, 1:] - adapted[:, :-1]
        difference = torch.cat((torch.zeros_like(difference[:, :1]), difference), dim=1)
        return difference.clamp_min(0), (-difference).clamp_min(0)
