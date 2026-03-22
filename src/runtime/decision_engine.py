"""Source-neutral runtime decision engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.runtime.adaptation import AdaptationLayer
from src.runtime.baseline import BaselineInference
from src.runtime.constants import CYTON_CHANNELS, CYTON_SAMPLE_RATE, WINDOW_SEC
from src.runtime.contracts import DecisionOutput, EEGChunk
from src.runtime.postprocessor import DecisionPostprocessor
from src.runtime.router import SourceRouter
from src.runtime.window_buffer import WindowBuffer


@dataclass(slots=True)
class WarmupSummary:
    """Simple warmup statistics for live sessions before runtime decisions begin."""

    duration_seconds: float
    sample_rate: float
    channel_names: list[str]
    mean: np.ndarray
    std: np.ndarray
    quality: dict[str, Any]


class DecisionEngine:
    """Router -> EEG stream -> adaptation -> baseline -> postprocessor."""

    def __init__(
        self,
        *,
        router: SourceRouter | None = None,
        adaptation: AdaptationLayer | None = None,
        baseline: BaselineInference | None = None,
        postprocessor: DecisionPostprocessor | None = None,
        buffer: WindowBuffer | None = None,
    ) -> None:
        self.router = router
        self.adaptation = adaptation or AdaptationLayer()
        self.baseline = baseline or BaselineInference()
        self.postprocessor = postprocessor or DecisionPostprocessor()
        self.buffer = buffer or WindowBuffer(
            channel_names=list(CYTON_CHANNELS),
            sample_rate=float(CYTON_SAMPLE_RATE),
            window_seconds=WINDOW_SEC,
        )

    def process_chunk(self, chunk: EEGChunk) -> list[DecisionOutput]:
        adapted = self.adaptation.transform(chunk)
        self.buffer.append(adapted.chunk)
        outputs: list[DecisionOutput] = []
        for window in self.buffer.pop_ready_windows():
            prediction = self.baseline.predict_with_details(
                window.data,
                sampling_rate=window.sample_rate,
                channel_names=window.channel_names,
            )
            outputs.append(
                self.postprocessor.update(
                    prediction.scores,
                    timestamp_start=window.timestamp_start,
                    timestamp_end=window.timestamp_end,
                    profile=self.adaptation.profile,
                    metadata={
                        **dict(window.metadata),
                        **dict(prediction.metadata),
                    },
                )
            )
        return outputs

    def step(self) -> DecisionOutput | None:
        if self.router is None:
            raise RuntimeError("DecisionEngine.step() requires a SourceRouter.")
        outputs = self.process_chunk(self.router.read_chunk())
        return outputs[-1] if outputs else None

    def warmup(self, duration_seconds: float = 10.0) -> WarmupSummary:
        if self.router is None:
            raise RuntimeError("DecisionEngine.warmup() requires a SourceRouter.")
        chunks: list[EEGChunk] = []
        accumulated = 0.0
        while accumulated < duration_seconds:
            chunk = self.router.read_chunk()
            chunks.append(chunk)
            accumulated += chunk.duration_seconds
        signal = np.concatenate([chunk.data for chunk in chunks], axis=1) if chunks else np.empty((0, 0), dtype=float)
        channel_names = list(chunks[0].channel_names) if chunks else list(CYTON_CHANNELS)
        sample_rate = float(chunks[0].sample_rate) if chunks else float(CYTON_SAMPLE_RATE)
        quality = {
            "chunk_count": len(chunks),
            "total_samples": int(signal.shape[1]) if signal.ndim == 2 else 0,
            "duration_seconds": float(accumulated),
        }
        return WarmupSummary(
            duration_seconds=float(accumulated),
            sample_rate=sample_rate,
            channel_names=channel_names,
            mean=np.mean(signal, axis=1) if signal.size else np.zeros(len(channel_names), dtype=float),
            std=np.std(signal, axis=1) if signal.size else np.zeros(len(channel_names), dtype=float),
            quality=quality,
        )
