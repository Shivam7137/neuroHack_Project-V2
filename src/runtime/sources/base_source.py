"""Base source contract for runtime EEG acquisition."""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.runtime.contracts import EEGChunk
from src.runtime.eeg_frame import EEGFrame, frame_from_chunk


class EEGSource(ABC):
    """Abstract canonical EEG source."""

    source_name = "unknown"

    @abstractmethod
    def start(self) -> None:
        """Start acquisition."""

    @abstractmethod
    def stop(self) -> None:
        """Stop acquisition."""

    @abstractmethod
    def read_chunk(self) -> EEGChunk:
        """Return the next source-neutral runtime chunk."""

    def read_frame(self) -> EEGFrame:
        """Return the next legacy runtime chunk with debug provenance."""
        return frame_from_chunk(self.read_chunk(), source=self.source_name)
