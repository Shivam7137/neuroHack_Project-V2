"""Teacher API over the current feature-based baseline models."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.inference.scorer import RuntimeScorer


class TeacherAPI:
    """Expose task scores and model-ready feature embeddings for generator training."""

    def __init__(
        self,
        artifacts_root: Path | None = None,
        calibration_path: Path | None = None,
        scorer: RuntimeScorer | None = None,
    ) -> None:
        self.scorer = scorer or RuntimeScorer(artifacts_root=artifacts_root, calibration_path=calibration_path)

    def predict_concentration(
        self,
        window: np.ndarray,
        sampling_rate: float,
        channel_names: list[str] | None = None,
    ) -> float:
        payload = self.scorer.score_task("concentration", window, sampling_rate, channel_names)
        return float(payload["concentration_probability"])

    def predict_stress(
        self,
        window: np.ndarray,
        sampling_rate: float,
        channel_names: list[str] | None = None,
    ) -> float:
        payload = self.scorer.score_task("stress", window, sampling_rate, channel_names)
        return float(payload["stress_score"] / 100.0)

    def extract_feature_embedding(
        self,
        window: np.ndarray,
        sampling_rate: float,
        channel_names: list[str] | None = None,
    ) -> np.ndarray:
        return self.scorer.extract_feature_embedding(window, sampling_rate, channel_names)

    def extract_task_feature_embedding(
        self,
        task_name: str,
        window: np.ndarray,
        sampling_rate: float,
        channel_names: list[str] | None = None,
    ) -> np.ndarray:
        return self.scorer.extract_task_feature_embedding(task_name, window, sampling_rate, channel_names)
