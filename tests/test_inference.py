"""Inference channel-handling tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.config import Settings
from src.inference.scorer import load_models, score_window
from src.preprocessing.normalization import DeterministicRecordingNormalizer
from src.training.common import PreprocessorBundle
from src.utils.io import save_pickle


def _write_fake_task_bundle(root: Path, task_name: str, profile_name: str) -> None:
    task_root = root / task_name
    task_root.mkdir(parents=True, exist_ok=True)
    preprocessor = PreprocessorBundle(
        task_name=task_name,
        profile_name=profile_name,
        channel_names=["C3"],
        raw_normalizer=DeterministicRecordingNormalizer(
            channel_names=["C3"],
            apply_recording_robust_scaling=False,
        ),
        bandpass_low=None,
        bandpass_high=None,
        notch_freq=None,
    )
    save_pickle(
        {
            "model": {"profile_name": profile_name},
            "model_name": f"{task_name}_{profile_name}",
            "prediction_mode": "classifier",
        },
        task_root / "model.pkl",
    )
    save_pickle(preprocessor, task_root / "preprocessor.pkl")


def test_score_window_reorders_channels(trained_artifacts: dict[str, object]) -> None:
    payload = np.load(trained_artifacts["eegmat_root"] / "Subject02_2.npz", allow_pickle=True)
    window = payload["signal"][[1, 0, 3, 2], :]
    result = score_window(
        window,
        sampling_rate=128.0,
        channel_names=["C4", "C3", "P4", "P3"],
        artifacts_root=trained_artifacts["artifacts_root"],
    )
    assert 0.0 <= result["concentration_probability"] <= 1.0


def test_score_window_fails_on_missing_channels(trained_artifacts: dict[str, object]) -> None:
    payload = np.load(trained_artifacts["eegmat_root"] / "Subject02_2.npz", allow_pickle=True)
    with pytest.raises(ValueError, match="Missing required channels"):
        score_window(
            payload["signal"][:3, :],
            sampling_rate=128.0,
            channel_names=["C3", "C4", "P3"],
            artifacts_root=trained_artifacts["artifacts_root"],
        )


def test_load_models_prefers_configured_cleanup_bundles(monkeypatch, tmp_path: Path) -> None:
    _write_fake_task_bundle(tmp_path, "concentration", "top_level_concentration")
    _write_fake_task_bundle(tmp_path, "stress", "top_level_stress")
    _write_fake_task_bundle(tmp_path / "cleanup_benchmark" / "light", "concentration", "light_concentration")
    _write_fake_task_bundle(tmp_path / "cleanup_benchmark" / "none", "stress", "none_stress")
    settings = Settings()
    settings.artifacts_root = tmp_path
    settings.concentration_cleanup_level = "light"
    settings.stress_cleanup_level = "none"
    monkeypatch.setattr("src.inference.scorer.get_settings", lambda: settings)

    models = load_models(tmp_path)

    assert models["concentration"].preprocessor.profile_name == "light_concentration"
    assert models["stress"].preprocessor.profile_name == "none_stress"
