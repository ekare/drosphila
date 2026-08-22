"""Per-sample rotation and selective-prediction metrics."""

from __future__ import annotations

import math
from typing import Iterable

import torch

from .losses import rotvec_to_matrix


def per_sample_geodesic_rad(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_matrix = rotvec_to_matrix(prediction)
    target_matrix = rotvec_to_matrix(target)
    relative = pred_matrix.transpose(-1, -2) @ target_matrix
    cosine = ((relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) / 2.0).clamp(-1.0, 1.0)
    return torch.acos(cosine)


def _percentile(values: torch.Tensor, percentile: float) -> float | None:
    if values.numel() == 0:
        return None
    return float(torch.quantile(values, percentile / 100.0).item())


def selective_risk(
    error: torch.Tensor,
    confidence: torch.Tensor,
    coverages: Iterable[float] = (0.25, 0.50, 0.75, 1.0),
) -> list[dict[str, float | int]]:
    """Return geodesic risk after retaining highest-confidence examples."""
    if error.ndim != 1 or confidence.ndim != 1 or error.shape != confidence.shape:
        raise ValueError("error and confidence must be equally shaped vectors")
    order = torch.argsort(confidence, descending=True)
    ranked = error[order]
    result = []
    for coverage in coverages:
        if not 0.0 < coverage <= 1.0:
            raise ValueError("coverage must be in (0, 1]")
        count = max(1, min(len(ranked), math.ceil(len(ranked) * coverage)))
        result.append(
            {
                "coverage": float(coverage),
                "samples": int(count),
                "risk_geodesic_deg": float(torch.rad2deg(ranked[:count]).mean().item()),
            }
        )
    return result


def rotation_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    confidence: torch.Tensor | None = None,
    min_axis_angle_rad: float = 0.02,
) -> dict:
    if prediction.shape != target.shape or prediction.ndim != 2 or prediction.shape[-1] != 3:
        raise ValueError("prediction and target must both have shape N,3")
    geodesic = per_sample_geodesic_rad(prediction, target)
    target_magnitude = torch.linalg.vector_norm(target, dim=-1)
    prediction_magnitude = torch.linalg.vector_norm(prediction, dim=-1)
    magnitude_error = (prediction_magnitude - target_magnitude).abs()
    axis_mask = target_magnitude >= min_axis_angle_rad
    if axis_mask.any():
        target_axis = target[axis_mask] / target_magnitude[axis_mask, None]
        safe_prediction = prediction[axis_mask] / prediction_magnitude[axis_mask, None].clamp_min(1e-8)
        axis_cosine = (target_axis * safe_prediction).sum(dim=-1).clamp(-1.0, 1.0)
        direction_accuracy = float((axis_cosine > 0.0).float().mean().item())
        axis_cosine_mean = float(axis_cosine.mean().item())
    else:
        direction_accuracy = None
        axis_cosine_mean = None
    result = {
        "samples": int(len(geodesic)),
        "model_mse": float((prediction - target).square().mean().item()),
        "zero_mse": float(target.square().mean().item()),
        "model_below_zero": bool((prediction - target).square().mean() < target.square().mean()),
        "mean_geodesic_deg": float(torch.rad2deg(geodesic).mean().item()),
        "median_geodesic_deg": float(torch.rad2deg(geodesic).median().item()),
        "p90_geodesic_deg": _percentile(torch.rad2deg(geodesic), 90.0),
        "component_mae_rad": (prediction - target).abs().mean(dim=0).tolist(),
        "component_mae_deg": torch.rad2deg((prediction - target).abs().mean(dim=0)).tolist(),
        "rotation_magnitude_mae_rad": float(magnitude_error.mean().item()),
        "rotation_magnitude_mae_deg": float(torch.rad2deg(magnitude_error).mean().item()),
        "axis_samples": int(axis_mask.sum().item()),
        "axis_cosine_mean": axis_cosine_mean,
        "direction_accuracy": direction_accuracy,
    }
    if confidence is not None:
        confidence = confidence.reshape(-1).to(geodesic)
        if confidence.shape != geodesic.shape:
            raise ValueError("confidence must have one value per prediction")
        result["confidence_mean"] = float(confidence.mean().item())
        result["confidence_std"] = float(confidence.std(unbiased=False).item())
        if confidence.std(unbiased=False) > 1e-8 and geodesic.std(unbiased=False) > 1e-8:
            result["confidence_error_correlation"] = float(
                torch.corrcoef(torch.stack((confidence, geodesic)))[0, 1].item()
            )
        else:
            result["confidence_error_correlation"] = None
        order = torch.argsort(confidence)
        result["confidence_bins_low_to_high"] = []
        for chunk in torch.tensor_split(order, 5):
            if len(chunk):
                result["confidence_bins_low_to_high"].append(
                    {
                        "samples": int(len(chunk)),
                        "mean_confidence": float(confidence[chunk].mean().item()),
                        "mean_geodesic_deg": float(torch.rad2deg(geodesic[chunk]).mean().item()),
                    }
                )
        result["selective_risk"] = selective_risk(geodesic, confidence)
    return result
