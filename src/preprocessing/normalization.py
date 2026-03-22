"""Deterministic normalization helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from sklearn.preprocessing import StandardScaler


def canonicalize_channel_name(name: str) -> str:
    """Normalize channel labels across datasets."""
    cleaned = name.replace("EEG ", "").replace("EEG", "").strip()
    legacy_map = {
        "T3": "T7",
        "T4": "T8",
        "T5": "P7",
        "T6": "P8",
    }
    return legacy_map.get(cleaned, cleaned)


def canonicalize_channel_names(channel_names: Sequence[str]) -> list[str]:
    """Normalize a channel list."""
    return [canonicalize_channel_name(name) for name in channel_names]


def align_signal_channels(
    signal: np.ndarray,
    channel_names: Sequence[str],
    target_channel_names: Sequence[str],
) -> np.ndarray:
    """Reorder or select channels to match a canonical order."""
    current = canonicalize_channel_names(channel_names)
    target = canonicalize_channel_names(target_channel_names)
    if not current:
        if signal.shape[0] != len(target):
            raise ValueError("Signal channel count does not match expected channel order.")
        return signal
    index_map = {name: idx for idx, name in enumerate(current)}
    missing = [name for name in target if name not in index_map]
    if missing:
        raise ValueError(f"Missing required channels: {missing}")
    indices = [index_map[name] for name in target]
    return signal[indices, :]


def robust_scale_per_channel(signal: np.ndarray) -> np.ndarray:
    """Apply deterministic robust scaling to one recording."""
    medians = np.median(signal, axis=1, keepdims=True)
    q75 = np.percentile(signal, 75, axis=1, keepdims=True)
    q25 = np.percentile(signal, 25, axis=1, keepdims=True)
    iqrs = np.maximum(q75 - q25, 1e-6)
    return (signal - medians) / iqrs


def average_rereference(signal: np.ndarray) -> np.ndarray:
    """Apply common average reference."""
    return signal - np.mean(signal, axis=0, keepdims=True)


def demean_window_channels(window: np.ndarray) -> np.ndarray:
    """Remove per-channel window means."""
    return window - np.mean(window, axis=1, keepdims=True)


@dataclass(slots=True)
class DeterministicRecordingNormalizer:
    """Deterministic recording-level preprocessing metadata."""

    channel_names: list[str] = field(default_factory=list)
    dropped_channels: list[str] = field(default_factory=list)
    rereference_mode: str = "none"
    apply_recording_robust_scaling: bool = True

    def transform(self, signal: np.ndarray, channel_names: Sequence[str] | None) -> np.ndarray:
        """Align and deterministically normalize one recording."""
        aligned = signal if channel_names is None else align_signal_channels(signal, channel_names, self.channel_names)
        if self.rereference_mode == "average":
            aligned = average_rereference(aligned)
        if self.apply_recording_robust_scaling:
            aligned = robust_scale_per_channel(aligned)
        return aligned


@dataclass(slots=True)
class FeatureScaler:
    """Feature scaler wrapper for models that require train-fit scaling."""

    scaler: StandardScaler | None = None

    def fit(self, features: np.ndarray) -> "FeatureScaler":
        """Fit a feature scaler."""
        self.scaler = StandardScaler()
        self.scaler.fit(features)
        return self

    def transform(self, features: np.ndarray) -> np.ndarray:
        """Apply feature scaling."""
        if self.scaler is None:
            return features
        return self.scaler.transform(features)


@dataclass(slots=True)
class RecordingRobustScaler:
    """Train-fit robust scaler for continuous EEG recordings."""

    channel_names: list[str] = field(default_factory=list)
    medians: np.ndarray | None = None
    iqrs: np.ndarray | None = None

    def fit(self, recordings: Sequence[np.ndarray]) -> "RecordingRobustScaler":
        """Fit per-channel robust statistics on aligned train recordings only."""
        if not recordings:
            raise ValueError("At least one recording is required to fit RecordingRobustScaler.")
        first = recordings[0]
        if first.ndim != 2:
            raise ValueError(f"Expected 2D recordings, received shape {first.shape}.")
        n_channels = first.shape[0]
        if self.channel_names and len(self.channel_names) != n_channels:
            raise ValueError("Configured channel_names do not match recording channel count.")
        for recording in recordings:
            if recording.ndim != 2 or recording.shape[0] != n_channels:
                raise ValueError("All recordings must share the same [channels, samples] shape convention.")
        stacked = np.concatenate(recordings, axis=1)
        self.medians = np.median(stacked, axis=1, keepdims=True)
        q75 = np.percentile(stacked, 75, axis=1, keepdims=True)
        q25 = np.percentile(stacked, 25, axis=1, keepdims=True)
        self.iqrs = np.maximum(q75 - q25, 1e-6)
        if not self.channel_names:
            self.channel_names = [f"ch_{idx:03d}" for idx in range(n_channels)]
        return self

    def transform(self, signal: np.ndarray, channel_names: Sequence[str] | None = None) -> np.ndarray:
        """Apply the fitted robust scaling after optional channel alignment."""
        if self.medians is None or self.iqrs is None:
            raise ValueError("RecordingRobustScaler must be fit before calling transform.")
        aligned = signal if channel_names is None else align_signal_channels(signal, channel_names, self.channel_names)
        if aligned.shape[0] != self.medians.shape[0]:
            raise ValueError("Signal channel count does not match fitted scaler statistics.")
        return (aligned - self.medians) / self.iqrs
