"""Inference channel-handling tests."""

from __future__ import annotations

import numpy as np
import pytest

from src.inference.scorer import score_window


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
