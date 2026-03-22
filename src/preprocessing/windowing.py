"""Window creation helpers."""

from __future__ import annotations

import numpy as np


def create_windows(
    eeg: np.ndarray,
    sampling_rate: float,
    window_seconds: float,
    stride_seconds: float,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Create overlapping fixed-length windows from one signal."""
    window_size = int(round(window_seconds * sampling_rate))
    stride = int(round(stride_seconds * sampling_rate))
    if window_size <= 0 or stride <= 0:
        raise ValueError("Window and stride must be positive.")
    if eeg.shape[-1] < window_size:
        return np.empty((0, eeg.shape[0], window_size), dtype=float), []

    windows: list[np.ndarray] = []
    bounds: list[tuple[int, int]] = []
    for start in range(0, eeg.shape[-1] - window_size + 1, stride):
        stop = start + window_size
        windows.append(eeg[:, start:stop])
        bounds.append((start, stop))
    return np.stack(windows, axis=0), bounds
