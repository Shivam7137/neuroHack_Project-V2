"""Frequency-domain EEG features."""

from __future__ import annotations

import numpy as np
from scipy.integrate import simpson
from scipy.signal import welch

BANDS: dict[str, tuple[float, float]] = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
}


def compute_bandpower_summary(
    channel: np.ndarray,
    sampling_rate: float,
) -> tuple[dict[str, float], dict[str, float], dict[str, float], dict[str, float]]:
    """Return absolute, log-absolute, relative, and ratio bandpower features."""
    nperseg = min(256, channel.shape[-1])
    freqs, psd = welch(channel, fs=sampling_rate, nperseg=nperseg)
    absolute: dict[str, float] = {}
    for band_name, (low, high) in BANDS.items():
        mask = (freqs >= low) & (freqs < high)
        band_value = float(simpson(psd[mask], freqs[mask]) if mask.any() else 0.0)
        absolute[band_name] = band_value
    total_power = float(sum(absolute.values())) or 1e-6
    relative = {name: value / total_power for name, value in absolute.items()}
    log_absolute = {name: float(np.log1p(value)) for name, value in absolute.items()}
    ratios = {
        "alpha_beta_ratio": absolute["alpha"] / max(absolute["beta"], 1e-6),
        "theta_beta_ratio": absolute["theta"] / max(absolute["beta"], 1e-6),
        "beta_alpha_ratio": absolute["beta"] / max(absolute["alpha"], 1e-6),
        "alpha_theta_ratio": absolute["alpha"] / max(absolute["theta"], 1e-6),
        "engagement_index": absolute["beta"] / max(absolute["alpha"] + absolute["theta"], 1e-6),
    }
    return absolute, log_absolute, relative, {key: float(value) for key, value in ratios.items()}


def compute_bandpower_features(channel: np.ndarray, sampling_rate: float) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    """Return log-absolute powers, relative powers, and ratio/index features."""
    _, log_absolute, relative, ratios = compute_bandpower_summary(channel, sampling_rate)
    return log_absolute, relative, ratios
