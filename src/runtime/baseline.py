"""Baseline inference wrapper for runtime EEG windows."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src.inference.scorer import RuntimeScorer
from src.runtime.constants import CYTON_CHANNELS, CYTON_SAMPLE_RATE
from src.runtime.contracts import DecisionScores


@dataclass(slots=True)
class BaselinePrediction:
    """Runtime baseline scores plus raw scorer details for compatibility wrappers."""

    scores: DecisionScores
    metadata: dict[str, Any] = field(default_factory=dict)


class BaselineInference:
    """Wrap the trained baseline scorer behind a stable runtime API."""

    def __init__(
        self,
        scorer: RuntimeScorer | None = None,
        *,
        artifacts_root: Path | None = None,
        calibration_path: Path | None = None,
    ) -> None:
        self.scorer = scorer or RuntimeScorer(artifacts_root=artifacts_root, calibration_path=calibration_path)

    def predict(
        self,
        window: np.ndarray,
        sampling_rate: float = CYTON_SAMPLE_RATE,
        channel_names: list[str] | None = None,
    ) -> DecisionScores:
        return self.predict_with_details(window, sampling_rate, channel_names).scores

    def predict_with_details(
        self,
        window: np.ndarray,
        sampling_rate: float = CYTON_SAMPLE_RATE,
        channel_names: list[str] | None = None,
    ) -> BaselinePrediction:
        resolved_channel_names = list(channel_names or CYTON_CHANNELS)
        result = self.scorer.score_window(window, sampling_rate=sampling_rate, channel_names=resolved_channel_names)
        quality = float(np.clip(result["quality_score"] / 100.0, 0.0, 1.0))
        scores = DecisionScores(
            concentration=float(np.clip(result["concentration_probability"], 0.0, 1.0)),
            stress=float(np.clip(result["stress_score"] / 100.0, 0.0, 1.0)),
            confidence=quality,
            quality=quality,
        )
        metadata = {
            "concentration_score_100": float(result["concentration_score"]),
            "concentration_probability": float(result["concentration_probability"]),
            "stress_score_100": float(result["stress_score"]),
            "stress_predicted_class": str(result["stress_predicted_class"]),
            "quality_score_100": float(result["quality_score"]),
            "quality_label": str(result["quality_label"]),
            "artifact_probabilities": {key: float(value) for key, value in result["artifact_probabilities"].items()},
            "artifact_flags": dict(result["artifact_flags"]),
        }
        return BaselinePrediction(scores=scores, metadata=metadata)
