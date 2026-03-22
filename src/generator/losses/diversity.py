"""Diversity regularization for generated EEG batches."""

from __future__ import annotations

try:
    import torch
except ImportError:  # pragma: no cover - exercised by optional dependency checks
    torch = None


def diversity_loss(samples):
    if torch is None:  # pragma: no cover
        raise ImportError("torch is required for generator losses.")
    if samples.shape[0] < 2:
        return samples.new_tensor(0.0)
    flattened = samples.reshape(samples.shape[0], -1)
    distances = torch.cdist(flattened, flattened, p=2)
    mask = ~torch.eye(samples.shape[0], dtype=torch.bool, device=samples.device)
    return -torch.mean(distances[mask])
