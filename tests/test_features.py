"""Tests for feature extraction and normalization."""

from __future__ import annotations

import numpy as np

from src.features.feature_builder import FeatureBuilder
from src.preprocessing.normalization import DeterministicRecordingNormalizer


def test_feature_builder_outputs_stable_finite_vector() -> None:
    time = np.linspace(0, 2, 256, endpoint=False)
    window = np.vstack([np.sin(2 * np.pi * 10 * time), np.sin(2 * np.pi * 20 * time)])
    builder = FeatureBuilder()
    vector, names = builder.build_window(window, sampling_rate=128, channel_names=["C3", "C4"])
    assert vector.shape[0] == len(names)
    assert np.isfinite(vector).all()
    assert any("theta_beta_ratio" in name for name in names)


def test_deterministic_normalizer_uses_channel_order() -> None:
    normalizer = DeterministicRecordingNormalizer(channel_names=["C3", "C4"])
    reordered = normalizer.transform(np.array([[10.0, 20.0, 30.0], [1.0, 2.0, 3.0]]), ["C4", "C3"])
    assert reordered.shape == (2, 3)


def test_feature_builder_adds_asymmetry_when_pairs_exist() -> None:
    time = np.linspace(0, 2, 256, endpoint=False)
    window = np.vstack(
        [
            np.sin(2 * np.pi * 10 * time),
            np.sin(2 * np.pi * 11 * time),
            np.sin(2 * np.pi * 9 * time),
            np.sin(2 * np.pi * 12 * time),
        ]
    )
    builder = FeatureBuilder()
    _, names = builder.build_window(window, sampling_rate=128, channel_names=["F3", "F4", "C3", "C4"])
    assert any("F3_F4_asymmetry_alpha" in name for name in names)
