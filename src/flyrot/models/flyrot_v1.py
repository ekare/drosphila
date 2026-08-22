"""Experimental FlyRot-v1 with separated polarity paths and SO(3) memory."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from flyrot.diagnostics import build_rotation_residual_diagnostics
from flyrot.losses import rotvec_to_matrix

from .accumulator import CausalEvidenceAccumulator
from .direction_cells import DirectionCellBank
from .photoreceptor import Photoreceptor
from .rotation_evidence import RotationEvidence


def _matrix_to_rotvec(rotation: torch.Tensor) -> torch.Tensor:
    """Differentiable SO(3) logarithm for matrices away from pi."""

    # Clamp away from the acos endpoints so the zero-initialized readout has
    # finite gradients during the first training step.
    cosine = ((rotation.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) / 2.0).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    angle = torch.acos(cosine)
    vee = torch.stack(
        (rotation[..., 2, 1] - rotation[..., 1, 2], rotation[..., 0, 2] - rotation[..., 2, 0], rotation[..., 1, 0] - rotation[..., 0, 1]),
        dim=-1,
    ) * 0.5
    sine = torch.linalg.vector_norm(vee, dim=-1)
    scale = torch.where(sine < 1e-5, 1.0 + angle.square() / 6.0, angle / sine.clamp_min(1e-8))
    return vee * scale.unsqueeze(-1)


def compose_step_rotations(step_rotation_vector: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Compose ordered step rotations and return matrices plus log vectors."""

    if step_rotation_vector.ndim != 3 or step_rotation_vector.shape[-1] != 3:
        raise ValueError("expected step rotations with shape B,T,3")
    step_matrices = rotvec_to_matrix(step_rotation_vector)
    state = torch.eye(3, dtype=step_rotation_vector.dtype, device=step_rotation_vector.device)
    state = state.expand(step_rotation_vector.shape[0], 3, 3).clone()
    cumulative = []
    for step in range(step_matrices.shape[1]):
        state = state @ step_matrices[:, step]
        cumulative.append(state)
    matrices = torch.stack(cumulative, dim=1)
    return matrices, _matrix_to_rotvec(matrices)


class FlyRotV1(nn.Module):
    """Small experimental retina with explicit polarity and orientation memory.

    This is deliberately a separate class from :class:`FlyRotV0`. Its
    ``rotation_vector`` output is the cumulative SO(3) composition, while
    ``step_rotation_vector`` exposes the per-pair estimate used by the step
    supervision path.
    """

    def __init__(
        self,
        scales: tuple[int, ...] = (4, 8, 16),
        field_size: int = 8,
        field_gain: float = 0.1,
    ) -> None:
        super().__init__()
        if field_size < 1 or field_gain < 0:
            raise ValueError("field_size must be positive and field_gain non-negative")
        self.scales = tuple(scales)
        self.field_size = field_size
        self.field_gain = field_gain
        self.photoreceptor = Photoreceptor()
        self.direction_cells = DirectionCellBank(scales=self.scales)
        self.on_rotation_evidence = RotationEvidence(scales=len(self.scales))
        self.off_rotation_evidence = RotationEvidence(scales=len(self.scales))
        self.on_accumulator = CausalEvidenceAccumulator(channels=3)
        self.off_accumulator = CausalEvidenceAccumulator(channels=3)
        field_features = 2 * field_size * field_size
        self.field_gate = nn.Linear(field_features, 6)
        self.step_readout = nn.Linear(6, 3)
        self.log_variance = nn.Parameter(torch.zeros(3))
        nn.init.zeros_(self.field_gate.weight)
        nn.init.zeros_(self.field_gate.bias)
        nn.init.zeros_(self.step_readout.weight)
        nn.init.zeros_(self.step_readout.bias)

    def forward(self, frames: torch.Tensor, diagnostics: bool = False) -> dict[str, torch.Tensor]:
        on, off = self.photoreceptor(frames)
        energy, valid, on_energy, off_energy = self.direction_cells(on, off, return_components=True)
        on_evidence = self.on_rotation_evidence(on_energy, valid)
        off_evidence = self.off_rotation_evidence(off_energy, valid)
        on_state = self.on_accumulator(on_evidence)
        off_state = self.off_accumulator(off_evidence)
        polarity_state = torch.cat((on_state, off_state), dim=-1)

        batch, steps, channels, height, width = on_energy.shape
        field = torch.cat((on_energy, off_energy), dim=2).reshape(batch * steps, -1, height, width)
        field = F.adaptive_avg_pool2d(field, (self.field_size, self.field_size))
        field = field.reshape(batch, steps, 2, self.direction_cells.channels_per_polarity, self.field_size, self.field_size)
        retinotopic_field = field.mean(dim=3)
        field_summary = retinotopic_field.reshape(batch, steps, -1)
        fused_state = polarity_state + self.field_gain * torch.tanh(self.field_gate(field_summary))
        step_rotation = self.step_readout(fused_state)
        cumulative_matrix, cumulative_rotation = compose_step_rotations(step_rotation)
        log_variance = self.log_variance.view(1, 1, 3).expand_as(step_rotation)
        confidence = torch.sigmoid(-log_variance.mean(dim=-1, keepdim=True))
        output = {
            "rotation_vector": cumulative_rotation,
            "rotation_matrix": cumulative_matrix,
            "step_rotation_vector": step_rotation,
            "step_rotation_matrix": rotvec_to_matrix(step_rotation),
            "log_variance": log_variance,
            "confidence": confidence,
            "direction_energy": energy,
            "valid_mask": valid,
            "on_motion_energy": on_energy,
            "off_motion_energy": off_energy,
            "on_rotation_evidence": on_evidence,
            "off_rotation_evidence": off_evidence,
            "retinotopic_field": retinotopic_field,
            "polarity_state": polarity_state,
        }
        if diagnostics:
            diagnostic_result = build_rotation_residual_diagnostics(
                cumulative_rotation,
                energy,
                valid,
                on_energy=on_energy,
                off_energy=off_energy,
            )
            output.update(diagnostic_result.as_dict())
            output["old_confidence"] = confidence
        return output
