"""Minimal FlyRot-v0 model wiring."""

from __future__ import annotations

import torch
from torch import nn

from .accumulator import CausalEvidenceAccumulator
from .direction_cells import DirectionCellBank
from .photoreceptor import Photoreceptor
from .rotation_evidence import RotationEvidence


class FlyRotV0(nn.Module):
    def __init__(
        self,
        scales: tuple[int, ...] = (1, 2),
        confidence_gated: bool = False,
        readout_scale: float = 1.0,
        magnitude_aware: bool = False,
        appearance_normalized: bool = False,
        focal_y_over_x: float = 1.0,
        scale_separated: bool = False,
        least_squares_basis: bool = False,
        motion_consistency_gate: bool = False,
        motion_consistency_floor: float = 0.25,
        input_dependent_uncertainty: bool = False,
        magnitude_confidence: bool = False,
        magnitude_confidence_center: float = 0.0103562189,
        magnitude_confidence_scale: float = 0.0052184332,
    ) -> None:
        super().__init__()
        self.confidence_gated = confidence_gated
        self.magnitude_aware = magnitude_aware
        self.scale_separated = scale_separated
        self.motion_consistency_gate = motion_consistency_gate
        if not 0.0 <= motion_consistency_floor <= 1.0:
            raise ValueError("motion_consistency_floor must be between 0 and 1")
        self.motion_consistency_floor = motion_consistency_floor
        self.input_dependent_uncertainty = input_dependent_uncertainty
        self.magnitude_confidence = magnitude_confidence
        if magnitude_confidence_scale <= 0:
            raise ValueError("magnitude_confidence_scale must be positive")
        self.magnitude_confidence_center = magnitude_confidence_center
        self.magnitude_confidence_scale = magnitude_confidence_scale
        self.photoreceptor = Photoreceptor()
        self.direction_cells = DirectionCellBank(scales=scales)
        self.rotation_evidence = RotationEvidence(
            scales=len(scales),
            appearance_normalized=appearance_normalized,
            focal_y_over_x=focal_y_over_x,
            scale_separated=scale_separated,
            least_squares_basis=least_squares_basis,
        )
        evidence_channels = (3 * len(scales) if scale_separated else 3)
        if magnitude_aware:
            evidence_channels += len(scales) if scale_separated else 1
        self.accumulator = CausalEvidenceAccumulator(channels=evidence_channels)
        self.rotation_readout = nn.Linear(evidence_channels, 3, bias=not confidence_gated)
        with torch.no_grad():
            self.rotation_readout.weight.zero_()
            self.rotation_readout.weight[:, :3].copy_(torch.eye(3) * readout_scale)
            if self.rotation_readout.bias is not None:
                self.rotation_readout.bias.zero_()
        if confidence_gated:
            self.gate_bias = nn.Parameter(torch.tensor(-1.0))
            self.gate_scale = nn.Parameter(torch.tensor(4.0))
        self.uncertainty_bias = nn.Parameter(torch.zeros(3))
        if input_dependent_uncertainty:
            self.uncertainty_readout = nn.Linear(evidence_channels, 3, bias=False)
            nn.init.zeros_(self.uncertainty_readout.weight)

    def forward(self, frames: torch.Tensor) -> dict[str, torch.Tensor]:
        on, off = self.photoreceptor(frames)
        energy, valid = self.direction_cells(on, off)
        evidence = self.rotation_evidence(energy, valid)
        if self.magnitude_aware:
            if self.scale_separated:
                motion_energy = energy.reshape(energy.shape[0], energy.shape[1], len(self.direction_cells.scales), -1)
                motion_energy = motion_energy.mean(dim=3).clamp_min(0)
            else:
                motion_energy = energy.mean(dim=(2, 3, 4)).clamp_min(0).unsqueeze(-1)
            motion_energy = torch.log1p(10.0 * motion_energy)
            evidence = torch.cat((evidence, motion_energy), dim=-1)
        accumulated_state = self.accumulator(evidence)
        accumulated = self.rotation_readout(accumulated_state)
        if self.motion_consistency_gate:
            coherence = self.rotation_evidence.rotational_coherence(energy, valid)
            consistency_gate = self.motion_consistency_floor + (1.0 - self.motion_consistency_floor) * coherence
            accumulated = accumulated * consistency_gate
        else:
            coherence = torch.ones_like(accumulated[..., :1])
        if self.confidence_gated:
            strength = accumulated_state.abs().mean(dim=-1, keepdim=True)
            gate = torch.sigmoid(self.gate_bias + self.gate_scale * strength * 10.0)
            accumulated = accumulated * gate
        else:
            gate = torch.ones_like(accumulated[..., :1])
        log_variance = self.uncertainty_bias.view(1, 1, 3).expand_as(accumulated)
        if self.input_dependent_uncertainty:
            log_variance = log_variance + self.uncertainty_readout(accumulated_state)
        if self.magnitude_confidence:
            prediction_magnitude = torch.linalg.vector_norm(accumulated, dim=-1, keepdim=True)
            confidence = torch.sigmoid(
                (prediction_magnitude - self.magnitude_confidence_center) / self.magnitude_confidence_scale
            )
        else:
            confidence = gate * torch.sigmoid(-log_variance.mean(dim=-1, keepdim=True))
        return {
            "rotation_vector": accumulated,
            "log_variance": log_variance,
            "confidence": confidence,
            "motion_coherence": coherence,
            "direction_energy": energy,
            "valid_mask": valid,
        }
