"""Small generic temporal-convolution comparison model."""

from __future__ import annotations

import torch
from torch import nn


class TinyConvBaseline(nn.Module):
    def __init__(self, channels: int = 8) -> None:
        super().__init__()
        self.spatial = nn.Sequential(
            nn.Conv2d(1, channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.readout = nn.Linear(channels, 3)
        self.log_variance = nn.Parameter(torch.zeros(3))

    def forward(self, frames: torch.Tensor) -> dict[str, torch.Tensor]:
        if frames.ndim != 5:
            raise ValueError("expected B,T,C,H,W input")
        if frames.shape[2] == 3:
            gray = 0.299 * frames[:, :, 0:1] + 0.587 * frames[:, :, 1:2] + 0.114 * frames[:, :, 2:3]
        elif frames.shape[2] == 1:
            gray = frames
        else:
            raise ValueError("frames must have one or three channels")
        delta = gray[:, 1:] - gray[:, :-1]
        b, t, _, h, w = delta.shape
        pooled = self.spatial(delta.reshape(b * t, 1, h, w)).flatten(1)
        rotation = self.readout(pooled).reshape(b, t, 3)
        log_variance = self.log_variance.view(1, 1, 3).expand_as(rotation)
        confidence = torch.exp(-log_variance).mean(dim=-1, keepdim=True).clamp(0, 1)
        return {"rotation_vector": rotation, "log_variance": log_variance, "confidence": confidence}
