"""Legacy runtime bridge layered on the source-neutral decision stack."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src.config import get_settings
from src.inference.scorer import RuntimeScorer
from src.runtime.adaptation import AdaptationLayer
from src.runtime.baseline import BaselineInference
from src.runtime.eeg_frame import EEGFrame
from src.runtime.postprocessor import DecisionPostprocessor
from src.runtime.treatment import TreatmentShim
from src.runtime.window_buffer import WindowBuffer


@dataclass(slots=True)
class EngineOutput:
    """One scored runtime window."""

    timestamp: float
    source: str
    sample_rate: float
    channel_names: list[str]
    concentration_score: float
    concentration_probability: float
    stress_score: float
    stress_predicted_class: str
    quality_score: float
    quality_label: str
    artifact_probabilities: dict[str, float]
    artifact_flags: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)


class StreamingEngine:
    """Compatibility wrapper that preserves the old runtime API."""

    def __init__(
        self,
        scorer: RuntimeScorer | None = None,
        treatment: TreatmentShim | None = None,
        buffer: WindowBuffer | None = None,
        compatibility_adapter: object | None = None,
        artifacts_root: Path | None = None,
        calibration_path: Path | None = None,
        adaptation: AdaptationLayer | None = None,
        baseline: BaselineInference | None = None,
        postprocessor: DecisionPostprocessor | None = None,
    ) -> None:
        settings = get_settings()
        self.settings = settings
        self.scorer = scorer or RuntimeScorer(artifacts_root=artifacts_root, calibration_path=calibration_path)
        self.treatment = treatment or TreatmentShim()
        self.compatibility_adapter = compatibility_adapter
        self.adaptation = adaptation or AdaptationLayer(
            canonical_channel_names=list(settings.cyton_channel_names),
            sample_rate=250.0,
        )
        self.baseline = baseline or BaselineInference(scorer=self.scorer)
        self.postprocessor = postprocessor or DecisionPostprocessor()
        self.buffer = buffer or WindowBuffer(
            channel_names=list(settings.cyton_channel_names),
            sample_rate=250.0,
            window_seconds=settings.runtime_window_seconds,
            stride_seconds=settings.runtime_stride_seconds,
            buffer_seconds=settings.runtime_buffer_seconds,
        )

    def _ensure_runtime_shape(self, frame: EEGFrame) -> None:
        """Lock default adaptation and buffering to the first observed stream shape."""
        if self.buffer.data.shape[1] > 0:
            if list(frame.channel_names) != self.buffer.channel_names or not np.isclose(frame.sample_rate, self.buffer.sample_rate):
                raise ValueError("Runtime stream shape changed after buffering began.")
            return
        if list(frame.channel_names) == self.buffer.channel_names and np.isclose(frame.sample_rate, self.buffer.sample_rate):
            return
        self.adaptation = AdaptationLayer(
            profile=self.adaptation.profile,
            canonical_channel_names=list(frame.channel_names),
            sample_rate=float(frame.sample_rate),
        )
        self.buffer = WindowBuffer(
            channel_names=list(frame.channel_names),
            sample_rate=float(frame.sample_rate),
            window_seconds=self.settings.runtime_window_seconds,
            stride_seconds=self.settings.runtime_stride_seconds,
            buffer_seconds=self.settings.runtime_buffer_seconds,
        )

    def process_frame(self, frame: EEGFrame) -> list[EngineOutput]:
        self._ensure_runtime_shape(frame)
        chunk = frame.to_chunk()
        chunk.metadata["source_name"] = frame.source
        chunk.metadata["legacy_frame_quality"] = dict(frame.quality)
        adapted = self.adaptation.transform(chunk)
        self.buffer.append(adapted.chunk)
        outputs: list[EngineOutput] = []
        for buffered_window in self.buffer.pop_ready_windows():
            treated = self.treatment.transform(
                buffered_window.data,
                sampling_rate=buffered_window.sample_rate,
                channel_names=buffered_window.channel_names,
            )
            prediction = self.baseline.predict_with_details(
                treated.window,
                sampling_rate=buffered_window.sample_rate,
                channel_names=buffered_window.channel_names,
            )
            decision = self.postprocessor.update(
                prediction.scores,
                timestamp_start=buffered_window.timestamp_start,
                timestamp_end=buffered_window.timestamp_end,
                profile=self.adaptation.profile,
                metadata={
                    **dict(buffered_window.metadata),
                    **dict(prediction.metadata),
                    "treatment_quality": dict(treated.quality),
                },
            )
            source_name = str(buffered_window.metadata.get("source_name", frame.source))
            outputs.append(
                EngineOutput(
                    timestamp=buffered_window.timestamp_end,
                    source=source_name,
                    sample_rate=buffered_window.sample_rate,
                    channel_names=list(buffered_window.channel_names),
                    concentration_score=float(decision.metadata["concentration_score_100"]),
                    concentration_probability=float(decision.metadata["concentration_probability"]),
                    stress_score=float(decision.metadata["stress_score_100"]),
                    stress_predicted_class=str(decision.metadata["stress_predicted_class"]),
                    quality_score=float(decision.metadata["quality_score_100"]),
                    quality_label=str(decision.metadata["quality_label"]),
                    artifact_probabilities={key: float(value) for key, value in decision.metadata["artifact_probabilities"].items()},
                    artifact_flags=dict(decision.metadata["artifact_flags"]),
                    metadata={
                        "decision_state": decision.state,
                        "concentration_smoothed": decision.concentration_smoothed,
                        "stress_smoothed": decision.stress_smoothed,
                        "confidence": decision.confidence,
                        "quality": decision.quality,
                        "adaptation_quality": adapted.quality,
                        "treatment_quality": dict(treated.quality),
                        "frame_quality": dict(frame.quality),
                    },
                )
            )
        return outputs

    def process_window(
        self,
        window: np.ndarray,
        sampling_rate: float,
        channel_names: list[str],
        source: str = "unknown",
        timestamp: float = 0.0,
    ) -> list[EngineOutput]:
        frame = EEGFrame(
            timestamp=timestamp,
            sample_rate=sampling_rate,
            channel_names=channel_names,
            data=window,
            source=source,
        )
        return self.process_frame(frame)
