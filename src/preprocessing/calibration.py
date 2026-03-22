"""Subject calibration profile helpers for future personalization hooks."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src.preprocessing.normalization import align_signal_channels
from src.utils.io import load_pickle, save_pickle


@dataclass(slots=True)
class CalibrationProfile:
    """Serializable per-subject calibration statistics."""

    subject_id: str | None
    session_id: str | None
    sampling_rate: float
    channel_names: list[str]
    medians: np.ndarray
    iqrs: np.ndarray
    channel_quality_weights: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    def transform(self, signal: np.ndarray, channel_names: list[str] | None) -> np.ndarray:
        """Apply aligned robust normalization and channel weighting."""
        aligned = signal if channel_names is None else align_signal_channels(signal, channel_names, self.channel_names)
        normalized = (aligned - self.medians) / np.maximum(self.iqrs, 1e-6)
        return normalized * self.channel_quality_weights


def build_calibration_profile(
    signal: np.ndarray,
    *,
    sampling_rate: float,
    channel_names: list[str],
    subject_id: str | None = None,
    session_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> CalibrationProfile:
    """Fit simple robust per-channel baseline statistics from a calibration recording."""
    medians = np.median(signal, axis=1, keepdims=True)
    q75 = np.percentile(signal, 75, axis=1, keepdims=True)
    q25 = np.percentile(signal, 25, axis=1, keepdims=True)
    iqrs = np.maximum(q75 - q25, 1e-6)
    channel_scale = np.median(iqrs)
    channel_quality_weights = np.clip(channel_scale / iqrs, 0.25, 4.0)
    return CalibrationProfile(
        subject_id=subject_id,
        session_id=session_id,
        sampling_rate=float(sampling_rate),
        channel_names=list(channel_names),
        medians=medians.astype(float),
        iqrs=iqrs.astype(float),
        channel_quality_weights=channel_quality_weights.astype(float),
        metadata=dict(metadata or {}),
    )


def save_calibration_profile(profile: CalibrationProfile, path: Path) -> None:
    """Persist one calibration profile to disk."""
    save_pickle(profile, path)


def load_calibration_profile(path: Path) -> CalibrationProfile:
    """Load one calibration profile from disk."""
    return load_pickle(path)
