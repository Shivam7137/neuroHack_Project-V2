"""Cleanup treatment profiles applied before feature extraction."""

from __future__ import annotations

import numpy as np
from scipy import ndimage, signal

CLEANUP_LEVELS = ("none", "light", "medium", "heavy")


def _channel_mad(eeg: np.ndarray) -> np.ndarray:
    medians = np.median(eeg, axis=1, keepdims=True)
    return np.maximum(np.median(np.abs(eeg - medians), axis=1, keepdims=True), 1e-6)


def _soft_clip(eeg: np.ndarray, clip_mad: float) -> np.ndarray:
    medians = np.median(eeg, axis=1, keepdims=True)
    limits = clip_mad * _channel_mad(eeg)
    return np.clip(eeg, medians - limits, medians + limits)


def _despike(eeg: np.ndarray, spike_mad: float, kernel_size: int) -> np.ndarray:
    local_median = ndimage.median_filter(eeg, size=(1, kernel_size), mode="nearest")
    residual = eeg - local_median
    mask = np.abs(residual) > (spike_mad * _channel_mad(residual))
    return np.where(mask, local_median, eeg)


def _smooth(eeg: np.ndarray, sampling_rate: float, window_fraction: float, blend: float) -> np.ndarray:
    window_length = max(5, int(round(sampling_rate * window_fraction)))
    if window_length % 2 == 0:
        window_length += 1
    if eeg.shape[-1] <= window_length:
        return eeg
    smoothed = signal.savgol_filter(eeg, window_length=window_length, polyorder=2, axis=-1, mode="interp")
    return ((1.0 - blend) * eeg) + (blend * smoothed)


def apply_cleanup_treatment(
    eeg: np.ndarray,
    *,
    sampling_rate: float,
    cleanup_level: str,
) -> np.ndarray:
    """Apply a cleanup treatment profile to an EEG signal."""
    level = cleanup_level.strip().lower()
    if level not in CLEANUP_LEVELS:
        raise ValueError(f"Unsupported cleanup level '{cleanup_level}'. Expected one of {CLEANUP_LEVELS}.")
    if level == "none":
        return eeg

    processed = _soft_clip(eeg, clip_mad=8.0 if level == "light" else 6.0 if level == "medium" else 4.5)
    processed = _despike(
        processed,
        spike_mad=7.0 if level == "light" else 5.5 if level == "medium" else 4.5,
        kernel_size=5 if level == "light" else 7 if level == "medium" else 9,
    )
    if level == "medium":
        processed = _smooth(processed, sampling_rate=sampling_rate, window_fraction=0.02, blend=0.12)
    if level == "heavy":
        processed = _smooth(processed, sampling_rate=sampling_rate, window_fraction=0.03, blend=0.22)
    return processed
