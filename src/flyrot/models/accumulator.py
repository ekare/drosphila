"""Small causal evidence accumulator."""

from __future__ import annotations

import torch
from torch import nn


class CausalEvidenceAccumulator(nn.Module):
    def __init__(self, channels: int = 3) -> None:
        super().__init__()
        self.leak_logit = nn.Parameter(torch.tensor(-1.0))
        self.state_scale = nn.Parameter(torch.ones(channels))

    def forward(self, evidence: torch.Tensor) -> torch.Tensor:
        if evidence.ndim != 3:
            raise ValueError("expected B,T,C evidence")
        leak = self.leak_logit.sigmoid()
        state = evidence.new_zeros((evidence.shape[0], evidence.shape[2]))
        states = []
        for step in range(evidence.shape[1]):
            state = leak * state + (1 - leak) * evidence[:, step] * self.state_scale
            states.append(state)
        return torch.stack(states, dim=1)
