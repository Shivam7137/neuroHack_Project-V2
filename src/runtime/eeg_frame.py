"""Legacy runtime EEG frame compatibility helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.runtime.contracts import EEGChunk


@dataclass(slots=True)
class EEGFrame:
    """Legacy runtime chunk with explicit source provenance."""

    timestamp: float
    sample_rate: float
    channel_names: list[str]
    data: np.ndarray
    quality: dict[str, float | int | str | bool | None] = field(default_factory=dict)
    source: str = "unknown"

    def to_chunk(self) -> EEGChunk:
        """Convert one legacy frame to the source-neutral runtime contract."""
        timestamp_start = float(self.timestamp)
        timestamp_end = float(timestamp_start + (self.data.shape[1] / self.sample_rate)) if self.sample_rate else timestamp_start
        metadata: dict[str, Any] = dict(self.quality)
        if self.source:
            metadata["source_name"] = self.source
        return EEGChunk(
            timestamp_start=timestamp_start,
            timestamp_end=timestamp_end,
            sample_rate=float(self.sample_rate),
            channel_names=list(self.channel_names),
            data=np.asarray(self.data, dtype=float),
            metadata=metadata,
        )


def frame_from_chunk(chunk: EEGChunk, source: str = "unknown") -> EEGFrame:
    """Convert one source-neutral chunk back to the legacy frame contract."""
    quality = {key: value for key, value in chunk.metadata.items() if key != "source_name"}
    return EEGFrame(
        timestamp=float(chunk.timestamp_start),
        sample_rate=float(chunk.sample_rate),
        channel_names=list(chunk.channel_names),
        data=np.asarray(chunk.data, dtype=float),
        quality=quality,
        source=str(chunk.metadata.get("source_name", source)),
    )
