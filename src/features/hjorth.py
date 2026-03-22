"""Hjorth parameter features."""

from __future__ import annotations

import numpy as np


def compute_hjorth_parameters(channel: np.ndarray) -> dict[str, float]:
    """Compute Hjorth activity, mobility, and complexity."""
    first_derivative = np.diff(channel)
    second_derivative = np.diff(first_derivative)
    var_zero = float(np.var(channel))
    var_d1 = float(np.var(first_derivative)) if first_derivative.size else 0.0
    var_d2 = float(np.var(second_derivative)) if second_derivative.size else 0.0
    activity = var_zero
    mobility = np.sqrt(var_d1 / max(var_zero, 1e-6))
    complexity = np.sqrt(var_d2 / max(var_d1, 1e-6)) / max(mobility, 1e-6)
    return {
        "activity": float(activity),
        "mobility": float(mobility),
        "complexity": float(complexity),
    }
