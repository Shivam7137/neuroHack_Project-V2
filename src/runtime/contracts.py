"""Source-neutral runtime contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


RuntimeMetadataValue = float | int | str | bool | None


@dataclass(slots=True)
class EEGChunk:
    """Canonical runtime EEG chunk."""

    timestamp_start: float
    timestamp_end: float
    sample_rate: float
    channel_names: list[str]
    data: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_samples(self) -> int:
        return int(self.data.shape[1])

    @property
    def n_channels(self) -> int:
        return int(self.data.shape[0])

    @property
    def duration_seconds(self) -> float:
        return float(self.n_samples / self.sample_rate) if self.sample_rate else 0.0


@dataclass(slots=True)
class DecisionScores:
    """Baseline model output for one scoreable EEG window."""

    concentration: float
    stress: float
    confidence: float
    quality: float


@dataclass(slots=True)
class DecisionOutput:
    """Stable postprocessed runtime decision."""

    timestamp_start: float
    timestamp_end: float
    concentration_raw: float
    stress_raw: float
    concentration_smoothed: float
    stress_smoothed: float
    state: str
    confidence: float
    quality: float
    metadata: dict[str, Any] = field(default_factory=dict)
