"""Runtime source router."""

from __future__ import annotations

from src.runtime.contracts import EEGChunk
from src.runtime.sources.base_source import EEGSource


class SourceRouter:
    """Select the active runtime source while keeping the engine source-agnostic."""

    def __init__(self, sources: dict[str, EEGSource]):
        self.sources = dict(sources)
        self.active_name: str | None = None
        self.active_source: EEGSource | None = None

    def set_active(self, name: str) -> None:
        if name not in self.sources:
            raise KeyError(f"Unknown EEG source: {name}")
        if self.active_name == name and self.active_source is not None:
            return
        if self.active_source is not None:
            self.active_source.stop()
        self.active_name = name
        self.active_source = self.sources[name]
        self.active_source.start()

    def stop(self) -> None:
        if self.active_source is None:
            return
        self.active_source.stop()
        self.active_source = None
        self.active_name = None

    def read_chunk(self) -> EEGChunk:
        if self.active_source is None or self.active_name is None:
            raise RuntimeError("No source selected")
        chunk = self.active_source.read_chunk()
        metadata = dict(chunk.metadata)
        metadata["source_name"] = self.active_name
        return EEGChunk(
            timestamp_start=float(chunk.timestamp_start),
            timestamp_end=float(chunk.timestamp_end),
            sample_rate=float(chunk.sample_rate),
            channel_names=list(chunk.channel_names),
            data=chunk.data,
            metadata=metadata,
        )
