"""Teacher-guided losses over outputs and feature embeddings."""

from __future__ import annotations

try:
    import torch
except ImportError:  # pragma: no cover - exercised by optional dependency checks
    torch = None


def _require_torch():
    if torch is None:  # pragma: no cover
        raise ImportError("torch is required for generator losses.")


def teacher_output_loss(
    concentration_prediction,
    concentration_target,
    stress_prediction,
    stress_target,
):
    _require_torch()
    return torch.mean(torch.abs(concentration_prediction - concentration_target)) + torch.mean(
        torch.abs(stress_prediction - stress_target)
    )


def teacher_feature_loss(feature_embedding, target_embedding):
    _require_torch()
    return torch.mean(torch.abs(feature_embedding - target_embedding))
