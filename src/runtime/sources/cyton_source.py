"""BrainFlow-backed live source for OpenBCI Cyton."""

from __future__ import annotations

import time

import numpy as np

from src.config import get_settings
from src.runtime.constants import CYTON_SAMPLE_RATE
from src.runtime.contracts import EEGChunk
from src.runtime.sources.base_source import EEGSource


class CytonSource(EEGSource):
    """Live Cyton source using BrainFlow."""

    source_name = "cyton"

    def __init__(
        self,
        serial_port: str | None = None,
        chunk_samples: int = 125,
    ) -> None:
        self.settings = get_settings()
        self.serial_port = serial_port or self.settings.cyton_serial_port
        self.chunk_samples = int(chunk_samples)
        self.board = None
        self.board_id = None
        self.sample_rate = float(CYTON_SAMPLE_RATE)
        self.eeg_channels: list[int] = []
        self.timestamp_channel: int | None = None
        self.channel_names = list(self.settings.cyton_channel_names)

    def start(self) -> None:
        try:
            from brainflow.board_shim import BoardIds, BoardShim, BrainFlowInputParams
        except ImportError as exc:
            raise ImportError("brainflow is required for CytonSource.") from exc
        params = BrainFlowInputParams()
        params.serial_port = self.serial_port
        self.board_id = BoardIds.CYTON_BOARD
        self.board = BoardShim(self.board_id, params)
        self.board.prepare_session()
        self.board.start_stream()
        self.sample_rate = float(BoardShim.get_sampling_rate(self.board_id))
        if not np.isclose(self.sample_rate, CYTON_SAMPLE_RATE):
            raise ValueError(f"Expected Cyton sample rate {CYTON_SAMPLE_RATE}, received {self.sample_rate}.")
        self.eeg_channels = list(BoardShim.get_eeg_channels(self.board_id))
        try:
            self.timestamp_channel = int(BoardShim.get_timestamp_channel(self.board_id))
        except Exception:
            self.timestamp_channel = None

    def stop(self) -> None:
        if self.board is None:
            return
        try:
            self.board.stop_stream()
        finally:
            self.board.release_session()
        self.board = None

    def read_chunk(self) -> EEGChunk:
        if self.board is None:
            raise RuntimeError("CytonSource.start() must be called before read_chunk().")
        deadline = time.monotonic() + 5.0
        while self.board.get_board_data_count() < self.chunk_samples:
            if time.monotonic() >= deadline:
                raise TimeoutError("Timed out waiting for a Cyton chunk from BrainFlow.")
            time.sleep(0.01)
        payload = self.board.get_board_data(self.chunk_samples)
        eeg = np.asarray(payload[self.eeg_channels, :], dtype=float)
        if self.timestamp_channel is not None:
            timestamps = np.asarray(payload[self.timestamp_channel, :], dtype=float)
            timestamp = float(timestamps[0])
        else:
            timestamp = time.time() - (self.chunk_samples / self.sample_rate)
        return EEGChunk(
            timestamp_start=timestamp,
            timestamp_end=float(timestamp + (eeg.shape[1] / self.sample_rate)),
            sample_rate=float(self.sample_rate),
            channel_names=list(self.channel_names),
            data=eeg,
            metadata={"board_data_count": self.chunk_samples},
        )
