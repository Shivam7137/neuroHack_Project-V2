"""Simple temporal EEG features."""

from __future__ import annotations

import numpy as np


def compute_temporal_stats(channel: np.ndarray) -> dict[str, float]:
    """Compute simple temporal statistics."""
    return {
        "mean": float(np.mean(channel)),
        "std": float(np.std(channel)),
        "rms": float(np.sqrt(np.mean(np.square(channel)))),
    }
