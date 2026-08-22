"""Project local direction-cell energy onto rotational optical-flow bases."""

from __future__ import annotations

import math

import torch
from torch import nn

from flyrot.geometry.rotational_flow import rotational_flow_basis


class RotationEvidence(nn.Module):
    def __init__(
        self,
        directions: int = 8,
        scales: int = 2,
        appearance_normalized: bool = False,
        focal_y_over_x: float = 1.0,
        scale_separated: bool = False,
        least_squares_basis: bool = False,
    ) -> None:
        super().__init__()
        self.directions = directions
        self.scales = scales
        self.appearance_normalized = appearance_normalized
        self.scale_separated = scale_separated
        self.least_squares_basis = least_squares_basis
        if scale_separated and least_squares_basis:
            raise ValueError("scale_separated and least_squares_basis are mutually exclusive")
        if focal_y_over_x <= 0:
            raise ValueError("focal_y_over_x must be positive")
        self.focal_y_over_x = focal_y_over_x
        self.gain = nn.Parameter(torch.ones(3))

    def forward(self, energy: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        if energy.ndim != 5:
            raise ValueError("expected energy shape B,T,C,H,W")
        b, t, channels, h, w = energy.shape
        expected = self.directions * self.scales
        if channels != expected:
            raise ValueError(f"expected {expected} direction channels, got {channels}")
        device, dtype = energy.device, energy.dtype
        y_pixels = torch.arange(h, device=device, dtype=dtype) - (h - 1) / 2
        x_pixels = torch.arange(w, device=device, dtype=dtype) - (w - 1) / 2
        yy, xx = torch.meshgrid(
            y_pixels / ((h / 2) * self.focal_y_over_x),
            x_pixels / (w / 2),
            indexing="ij",
        )
        bases = rotational_flow_basis(xx, yy)
        directions = torch.arange(self.directions, device=device, dtype=dtype) * (2 * math.pi / self.directions)
        direction_vectors = torch.stack((directions.cos(), directions.sin()), dim=1)
        directional_projection = torch.einsum("dp,ophw->dohw", direction_vectors, bases)
        basis = directional_projection.repeat(self.scales, 1, 1, 1)
        weighted = energy * valid
        if self.appearance_normalized:
            valid_count = valid.sum(dim=(3, 4), keepdim=True).clamp_min(1.0)
            spatial_baseline = weighted.sum(dim=(3, 4), keepdim=True) / valid_count
            weighted = (weighted - spatial_baseline).clamp_min(0) * valid
        if self.least_squares_basis:
            weighted_by_scale = weighted.reshape(b, t, self.scales, self.directions, h, w)
            motion = torch.einsum("btsdhw,dp->btphw", weighted_by_scale, direction_vectors)
            pixel_weight = weighted_by_scale.sum(dim=(2, 3))
            basis_pixels = bases.permute(2, 3, 1, 0).reshape(h * w, 2, 3)
            motion_pixels = motion.permute(0, 1, 3, 4, 2).reshape(b, t, h * w, 2)
            weight_pixels = pixel_weight.reshape(b, t, h * w)
            normal = torch.einsum("pij,btp,pik->btjk", basis_pixels, weight_pixels, basis_pixels)
            rhs = torch.einsum("pij,btp,btpi->btj", basis_pixels, weight_pixels, motion_pixels)
            identity = torch.eye(3, device=device, dtype=dtype).view(1, 1, 3, 3)
            solution = torch.linalg.solve(normal + 1e-3 * identity, rhs)
            return solution * self.gain.view(1, 1, 3)
        if self.scale_separated:
            weighted_by_scale = weighted.reshape(b, t, self.scales, self.directions, h, w)
            denominator = weighted_by_scale.sum(dim=(3, 4, 5)).clamp_min(1e-6)
            basis_by_scale = directional_projection.unsqueeze(0).expand(self.scales, -1, -1, -1, -1)
            evidence_by_scale = torch.einsum("btsdhw,sdohw->btso", weighted_by_scale, basis_by_scale)
            evidence_by_scale = evidence_by_scale / denominator.unsqueeze(-1)
            return (evidence_by_scale * self.gain.view(1, 1, 1, 3)).reshape(b, t, self.scales * 3)
        denominator = weighted.sum(dim=(2, 3, 4)).clamp_min(1e-6)
        evidence = torch.einsum("btchw,cohw->bto", weighted, basis) / denominator.unsqueeze(-1)
        return evidence * self.gain

    def rotational_coherence(self, energy: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """Return an RGB-only consistency score for a single rotational flow."""
        if energy.ndim != 5:
            raise ValueError("expected energy shape B,T,C,H,W")
        b, t, channels, h, w = energy.shape
        expected = self.directions * self.scales
        if channels != expected:
            raise ValueError(f"expected {expected} direction channels, got {channels}")
        device, dtype = energy.device, energy.dtype
        y_pixels = torch.arange(h, device=device, dtype=dtype) - (h - 1) / 2
        x_pixels = torch.arange(w, device=device, dtype=dtype) - (w - 1) / 2
        yy, xx = torch.meshgrid(
            y_pixels / ((h / 2) * self.focal_y_over_x),
            x_pixels / (w / 2),
            indexing="ij",
        )
        bases = rotational_flow_basis(xx, yy)
        angles = torch.arange(self.directions, device=device, dtype=dtype) * (2 * math.pi / self.directions)
        direction_vectors = torch.stack((angles.cos(), angles.sin()), dim=1)
        weighted = energy.clamp_min(0) * valid
        by_scale = weighted.reshape(b, t, self.scales, self.directions, h, w)
        motion = torch.einsum("btsdhw,dp->btphw", by_scale, direction_vectors)
        motion_pixels = motion.permute(0, 1, 3, 4, 2).reshape(b, t, h * w, 2)
        motion_magnitude = torch.linalg.vector_norm(motion_pixels, dim=-1)
        denominator = weighted.sum(dim=(2, 3, 4)).clamp_min(1e-6)
        directional_projection = torch.einsum("dp,ophw->dohw", direction_vectors, bases)
        basis = directional_projection.repeat(self.scales, 1, 1, 1)
        evidence = torch.einsum("btchw,cohw->bto", weighted, basis) / denominator.unsqueeze(-1)
        basis_pixels = bases.permute(2, 3, 1, 0).reshape(h * w, 2, 3)
        predicted = torch.einsum("pij,btj->btpi", basis_pixels, evidence)
        predicted_magnitude = torch.linalg.vector_norm(predicted, dim=-1)
        cosine = (motion_pixels * predicted).sum(dim=-1) / (motion_magnitude * predicted_magnitude + 1e-6)
        aligned = cosine.clamp_min(0.0)
        return (
            (aligned * motion_magnitude).sum(dim=-1) / (motion_magnitude.sum(dim=-1) + 1e-6)
        ).clamp(0.0, 1.0).unsqueeze(-1)
