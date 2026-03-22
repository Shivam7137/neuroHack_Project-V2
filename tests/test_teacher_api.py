"""Teacher API tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

from src.config import Settings
from src.baseline.teacher_api import TeacherAPI


def _canonical_window(signal: np.ndarray) -> tuple[np.ndarray, list[str]]:
    settings = Settings()
    channel_names = list(settings.cyton_channel_names)
    resampled = resample_poly(signal, up=250, down=128, axis=-1)
    canonical = np.zeros((8, resampled.shape[1]), dtype=float)
    canonical[2] = resampled[0]
    canonical[3] = resampled[1]
    canonical[4] = resampled[2]
    canonical[5] = resampled[3]
    canonical[0] = 0.5 * resampled[0]
    canonical[1] = 0.5 * resampled[1]
    canonical[6] = 0.5 * resampled[2]
    canonical[7] = 0.5 * resampled[3]
    return canonical, channel_names


def test_teacher_api_returns_stable_finite_outputs(trained_artifacts: dict[str, Path]) -> None:
    payload = np.load(trained_artifacts["eegmat_root"] / "Subject04_2.npz", allow_pickle=True)
    window, channel_names = _canonical_window(payload["signal"])
    teacher = TeacherAPI(artifacts_root=trained_artifacts["artifacts_root"])
    first_embedding = teacher.extract_feature_embedding(window, 250.0, channel_names)
    second_embedding = teacher.extract_feature_embedding(window, 250.0, channel_names)
    concentration = teacher.predict_concentration(window, 250.0, channel_names)
    stress = teacher.predict_stress(window, 250.0, channel_names)
    assert np.isfinite(first_embedding).all()
    assert np.allclose(first_embedding, second_embedding)
    assert 0.0 <= concentration <= 1.0
    assert 0.0 <= stress <= 1.0
