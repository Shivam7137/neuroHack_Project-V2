"""Compatibility helpers between canonical runtime data and model inputs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.preprocessing.normalization import align_signal_channels, canonicalize_channel_names


@dataclass(slots=True)
class AdaptedWindow:
    """Model-specific view of a canonical window."""

    data: np.ndarray
    sampling_rate: float
    channel_names: list[str]


class CompatibilityAdapter:
    """Select and reorder task-required channels from a canonical runtime window."""

    def __init__(self, canonical_channel_names: list[str]) -> None:
        self.canonical_channel_names = canonicalize_channel_names(canonical_channel_names)

    def validate_canonical(self, window: np.ndarray, channel_names: list[str] | None) -> list[str]:
        names = canonicalize_channel_names(channel_names or self.canonical_channel_names)
        if window.ndim != 2:
            raise ValueError(f"Expected a [channels, samples] window, received shape {window.shape}.")
        if len(names) != window.shape[0]:
            raise ValueError("Window channel count does not match the provided channel names.")
        missing = [name for name in self.canonical_channel_names if name not in names]
        if missing:
            raise ValueError(f"Missing canonical runtime channels: {missing}")
        return names

    def adapt_window(
        self,
        window: np.ndarray,
        sampling_rate: float,
        channel_names: list[str],
        required_channel_names: list[str],
    ) -> AdaptedWindow:
        validated_names = self.validate_canonical(window, channel_names)
        adapted = align_signal_channels(window, validated_names, required_channel_names)
        return AdaptedWindow(
            data=adapted,
            sampling_rate=float(sampling_rate),
            channel_names=canonicalize_channel_names(required_channel_names),
        )
