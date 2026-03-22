"""Sliding window ring buffer for canonical EEG chunks."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from src.runtime.constants import EXACT_SAMPLES_PER_STRIDE, STRIDE_SEC
from src.runtime.contracts import EEGChunk
from src.runtime.eeg_frame import EEGFrame


@dataclass(slots=True)
class BufferedWindow:
    """A scoreable window extracted from the ring buffer."""

    timestamp_start: float
    timestamp_end: float
    data: np.ndarray
    channel_names: list[str]
    sample_rate: float
    metadata: dict[str, float | int | str | bool | None | list[str] | dict[str, object]]

    @property
    def timestamp(self) -> float:
        """Legacy timestamp alias matching the prior buffer API."""
        return float(self.timestamp_end)

    @property
    def source(self) -> str:
        """Legacy source alias for compatibility-only callers."""
        return str(self.metadata.get("source_name", "unknown"))

    @property
    def quality(self) -> dict[str, float | int | str | bool | None | list[str] | dict[str, object]]:
        """Legacy quality alias for compatibility-only callers."""
        return dict(self.metadata)


class WindowBuffer:
    """Retain canonical EEG chunks and emit overlapping score windows."""

    def __init__(
        self,
        channel_names: list[str],
        sample_rate: float,
        window_seconds: float,
        stride_seconds: float | None = None,
        buffer_seconds: float = 10.0,
    ) -> None:
        self.channel_names = list(channel_names)
        self.sample_rate = float(sample_rate)
        self.window_samples = int(round(window_seconds * self.sample_rate))
        stride_seconds = STRIDE_SEC if stride_seconds is None else float(stride_seconds)
        exact_stride_samples = float(stride_seconds * self.sample_rate)
        if math.isclose(stride_seconds, STRIDE_SEC) and math.isclose(self.sample_rate, 250.0):
            exact_stride_samples = float(EXACT_SAMPLES_PER_STRIDE)
        self.stride_samples = int(math.floor(exact_stride_samples))
        self.fractional_stride_samples = float(exact_stride_samples - self.stride_samples)
        self.capacity_samples = int(round(buffer_seconds * self.sample_rate))
        if self.window_samples <= 0 or self.stride_samples <= 0 or self.capacity_samples < self.window_samples:
            raise ValueError("Invalid runtime window buffer configuration.")
        self.data = np.empty((len(channel_names), 0), dtype=float)
        self.timestamps = np.empty((0,), dtype=float)
        self.base_sample_index = 0
        self.next_window_start = 0
        self._stride_residual = 0.0
        self.metadata: dict[str, float | int | str | bool | None | list[str] | dict[str, object]] = {}

    def append(self, frame: EEGFrame | EEGChunk) -> None:
        chunk = frame.to_chunk() if isinstance(frame, EEGFrame) else frame
        if chunk.data.shape[0] != len(self.channel_names):
            raise ValueError("Frame channel count does not match buffer configuration.")
        if list(chunk.channel_names) != self.channel_names:
            raise ValueError("Frame channel ordering does not match buffer configuration.")
        if not np.isclose(chunk.sample_rate, self.sample_rate):
            raise ValueError("Frame sampling rate does not match buffer configuration.")
        sample_offsets = np.arange(chunk.data.shape[1], dtype=float) / self.sample_rate
        chunk_timestamps = chunk.timestamp_start + sample_offsets
        self.data = np.concatenate([self.data, chunk.data.astype(float, copy=False)], axis=1)
        self.timestamps = np.concatenate([self.timestamps, chunk_timestamps], axis=0)
        self.metadata = dict(chunk.metadata)
        self._trim()

    def _trim(self) -> None:
        overflow = self.data.shape[1] - self.capacity_samples
        if overflow <= 0:
            return
        self.data = self.data[:, overflow:]
        self.timestamps = self.timestamps[overflow:]
        self.base_sample_index += overflow
        if self.next_window_start < self.base_sample_index:
            self.next_window_start = self.base_sample_index

    def has_window(self, n_samples: int | None = None) -> bool:
        target = self.window_samples if n_samples is None else int(n_samples)
        return bool(self.data.shape[1] >= target)

    def latest_window(self, n_samples: int | None = None) -> np.ndarray:
        target = self.window_samples if n_samples is None else int(n_samples)
        if not self.has_window(target):
            raise ValueError("Requested window is not yet available.")
        return self.data[:, -target:].copy()

    def pop_ready_windows(self) -> list[BufferedWindow]:
        outputs: list[BufferedWindow] = []
        end_index = self.base_sample_index + self.data.shape[1]
        while self.next_window_start + self.window_samples <= end_index:
            local_start = self.next_window_start - self.base_sample_index
            local_stop = local_start + self.window_samples
            outputs.append(
                BufferedWindow(
                    timestamp_start=float(self.timestamps[local_start]),
                    timestamp_end=float(self.timestamps[local_stop - 1] + (1.0 / self.sample_rate)),
                    data=self.data[:, local_start:local_stop].copy(),
                    channel_names=list(self.channel_names),
                    sample_rate=self.sample_rate,
                    metadata=dict(self.metadata),
                )
            )
            stride_advance = self.stride_samples
            self._stride_residual += self.fractional_stride_samples
            if self._stride_residual >= 1.0:
                whole_samples = int(self._stride_residual)
                stride_advance += whole_samples
                self._stride_residual -= whole_samples
            self.next_window_start += stride_advance
        return outputs
