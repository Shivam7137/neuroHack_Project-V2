"""Window quality-control helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import welch


@dataclass(slots=True)
class WindowQualityConfig:
    """Deterministic window rejection thresholds."""

    flat_variance_threshold: float
    max_abs_amplitude_threshold: float
    max_variance_threshold: float
    max_line_noise_ratio: float
    line_noise_frequency: float | None


def assess_window_quality(
    window: np.ndarray,
    sampling_rate: float,
    config: WindowQualityConfig,
) -> tuple[bool, list[str]]:
    """Return whether a window is acceptable and rejection reasons."""
    reasons: list[str] = []
    channel_variances = np.var(window, axis=1)
    if np.any(channel_variances < config.flat_variance_threshold):
        reasons.append("flat_variance")
    if np.max(np.abs(window)) > config.max_abs_amplitude_threshold:
        reasons.append("extreme_amplitude")
    if np.max(channel_variances) > config.max_variance_threshold:
        reasons.append("extreme_variance")

    if config.line_noise_frequency is not None and config.line_noise_frequency > 0:
        nperseg = min(256, window.shape[-1])
        freqs, psd = welch(window, fs=sampling_rate, axis=-1, nperseg=nperseg)
        if not np.isfinite(psd).all():
            reasons.append("invalid_psd")
        total_power = np.sum(psd, axis=1)
        line_mask = np.abs(freqs - config.line_noise_frequency) <= 1.0
        if np.any(line_mask):
            line_power = np.sum(psd[:, line_mask], axis=1)
            ratios = line_power / np.maximum(total_power, 1e-6)
            if np.any(ratios > config.max_line_noise_ratio):
                reasons.append("line_noise")
    elif not np.isfinite(window).all():
        reasons.append("invalid_signal")

    return (len(reasons) == 0), reasons
