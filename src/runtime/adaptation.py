"""Neutral adaptation layer for source-agnostic runtime inference."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.preprocessing.normalization import canonicalize_channel_names
from src.runtime.constants import CYTON_CHANNELS, CYTON_SAMPLE_RATE
from src.runtime.contracts import EEGChunk


@dataclass(slots=True)
class AdaptationProfile:
    """Placeholder runtime profile for future calibration/personalization."""

    channel_scale: np.ndarray
    channel_bias: np.ndarray
    channel_weight: np.ndarray
    smoothing_alpha: float
    stress_threshold_shift: float
    concentration_threshold_shift: float
    enabled: bool = True


def default_profile(n_channels: int = len(CYTON_CHANNELS)) -> AdaptationProfile:
    """Return a neutral profile that leaves runtime behavior effectively unchanged."""
    return AdaptationProfile(
        channel_scale=np.ones(n_channels, dtype=float),
        channel_bias=np.zeros(n_channels, dtype=float),
        channel_weight=np.ones(n_channels, dtype=float),
        smoothing_alpha=0.0,
        stress_threshold_shift=0.0,
        concentration_threshold_shift=0.0,
        enabled=False,
    )


@dataclass(slots=True)
class AdaptationResult:
    """Adapted chunk plus generic runtime quality metadata."""

    chunk: EEGChunk
    quality: dict[str, Any] = field(default_factory=dict)


class AdaptationLayer:
    """Validate canonical runtime chunks and apply only neutral generic transforms by default."""

    def __init__(
        self,
        profile: AdaptationProfile | None = None,
        *,
        canonical_channel_names: list[str] | None = None,
        sample_rate: float = CYTON_SAMPLE_RATE,
        mask_bad_channels: bool = False,
        bad_channel_std_floor: float = 1e-8,
        max_abs_clip: float | None = None,
    ) -> None:
        self.profile = profile or default_profile()
        self.canonical_channel_names = canonicalize_channel_names(canonical_channel_names or CYTON_CHANNELS)
        self.sample_rate = float(sample_rate)
        self.mask_bad_channels = bool(mask_bad_channels)
        self.bad_channel_std_floor = float(bad_channel_std_floor)
        self.max_abs_clip = None if max_abs_clip is None else float(max_abs_clip)

    def transform(self, chunk: EEGChunk) -> AdaptationResult:
        """Return a source-neutral chunk with optional profile transforms and quality metadata."""
        channel_names = canonicalize_channel_names(chunk.channel_names)
        if chunk.data.ndim != 2:
            raise ValueError(f"Expected a [channels, samples] chunk, received shape {chunk.data.shape}.")
        if chunk.data.shape[0] != len(channel_names):
            raise ValueError("Chunk channel count does not match channel_names.")
        if channel_names != self.canonical_channel_names:
            raise ValueError(
                f"Chunk channel ordering does not match canonical runtime order. "
                f"Expected {self.canonical_channel_names}, received {channel_names}."
            )
        if not np.isclose(chunk.sample_rate, self.sample_rate):
            raise ValueError(
                f"Chunk sample rate does not match canonical runtime rate. "
                f"Expected {self.sample_rate}, received {chunk.sample_rate}."
            )

        data = np.asarray(chunk.data, dtype=float).copy()
        data[~np.isfinite(data)] = 0.0
        channel_std = np.std(data, axis=1)
        flat_channels = [channel_names[idx] for idx, value in enumerate(channel_std) if value <= self.bad_channel_std_floor]

        if self.mask_bad_channels and flat_channels:
            for channel_name in flat_channels:
                data[channel_names.index(channel_name), :] = 0.0

        if self.max_abs_clip is not None:
            data = np.clip(data, -self.max_abs_clip, self.max_abs_clip)

        if self.profile.enabled:
            data = (data * self.profile.channel_scale[:, None]) + self.profile.channel_bias[:, None]
            data = data * self.profile.channel_weight[:, None]

        rms = float(np.sqrt(np.mean(np.square(data)))) if data.size else 0.0
        max_abs = float(np.max(np.abs(data))) if data.size else 0.0
        quality = {
            "channel_count": int(data.shape[0]),
            "sample_count": int(data.shape[1]),
            "flat_channels": flat_channels,
            "masked_bad_channels": bool(self.mask_bad_channels and flat_channels),
            "rms": rms,
            "max_abs": max_abs,
            "profile_enabled": bool(self.profile.enabled),
        }
        metadata = dict(chunk.metadata)
        metadata["adaptation_quality"] = quality
        adapted = EEGChunk(
            timestamp_start=float(chunk.timestamp_start),
            timestamp_end=float(chunk.timestamp_end),
            sample_rate=float(chunk.sample_rate),
            channel_names=list(channel_names),
            data=data,
            metadata=metadata,
        )
        return AdaptationResult(chunk=adapted, quality=quality)
