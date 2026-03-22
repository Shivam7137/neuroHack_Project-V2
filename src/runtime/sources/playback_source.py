"""Playback source for canonical recordings stored on disk."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.config import get_settings
from src.runtime.constants import CYTON_SAMPLE_RATE
from src.runtime.contracts import EEGChunk
from src.runtime.sources.base_source import EEGSource


def _normalize_signal_shape(signal: np.ndarray) -> np.ndarray:
    if signal.ndim != 2:
        raise ValueError(f"Expected a 2D EEG array, received shape {signal.shape}.")
    return signal.astype(float) if signal.shape[0] <= signal.shape[1] else signal.T.astype(float)


class PlaybackSource(EEGSource):
    """Read canonical EEG chunks from a saved recording file."""

    source_name = "playback"

    def __init__(
        self,
        path: Path,
        chunk_samples: int = 125,
        source_label: str = "playback",
    ) -> None:
        self.path = Path(path)
        self.chunk_samples = int(chunk_samples)
        self.source_label = source_label
        self.source_name = source_label
        self.signal = np.empty((0, 0), dtype=float)
        self.sample_rate = 0.0
        self.channel_names: list[str] = []
        self.timestamps: np.ndarray | None = None
        self.cursor = 0

    def start(self) -> None:
        settings = get_settings()
        if self.path.suffix.lower() == ".npz":
            payload = np.load(self.path, allow_pickle=True)
            key = "signal" if "signal" in payload else "window" if "window" in payload else list(payload.keys())[0]
            self.signal = _normalize_signal_shape(np.asarray(payload[key], dtype=float))
            self.sample_rate = float(payload["sampling_rate"]) if "sampling_rate" in payload else float(CYTON_SAMPLE_RATE)
            if "channel_names" in payload:
                self.channel_names = [str(item) for item in payload["channel_names"].tolist()]
            elif self.signal.shape[0] == len(settings.cyton_channel_names):
                self.channel_names = list(settings.cyton_channel_names)
            else:
                self.channel_names = [f"ch_{idx:02d}" for idx in range(self.signal.shape[0])]
            self.timestamps = np.asarray(payload["timestamps"], dtype=float) if "timestamps" in payload else None
        elif self.path.suffix.lower() == ".npy":
            self.signal = _normalize_signal_shape(np.load(self.path))
            self.sample_rate = float(CYTON_SAMPLE_RATE)
            self.channel_names = list(settings.cyton_channel_names) if self.signal.shape[0] == len(settings.cyton_channel_names) else [f"ch_{idx:02d}" for idx in range(self.signal.shape[0])]
            self.timestamps = None
        elif self.path.suffix.lower() == ".csv":
            matrix = np.loadtxt(self.path, delimiter=",", dtype=float)
            self.signal = _normalize_signal_shape(matrix)
            self.sample_rate = float(CYTON_SAMPLE_RATE)
            self.channel_names = list(settings.cyton_channel_names) if self.signal.shape[0] == len(settings.cyton_channel_names) else [f"ch_{idx:02d}" for idx in range(self.signal.shape[0])]
            self.timestamps = None
        else:
            raise ValueError(f"Unsupported playback file format: {self.path.suffix}")
        self.cursor = 0

    def stop(self) -> None:
        self.cursor = self.signal.shape[1]

    def read_chunk(self) -> EEGChunk:
        if self.cursor >= self.signal.shape[1]:
            raise StopIteration("Playback recording has no more data.")
        stop = min(self.cursor + self.chunk_samples, self.signal.shape[1])
        chunk = self.signal[:, self.cursor:stop]
        if self.timestamps is not None and len(self.timestamps) >= stop:
            timestamp = float(self.timestamps[self.cursor])
        else:
            timestamp = float(self.cursor / self.sample_rate)
        self.cursor = stop
        return EEGChunk(
            timestamp_start=timestamp,
            timestamp_end=float(timestamp + (chunk.shape[1] / self.sample_rate)),
            sample_rate=float(self.sample_rate),
            channel_names=list(self.channel_names),
            data=chunk,
            metadata={"playback_cursor": self.cursor},
        )
