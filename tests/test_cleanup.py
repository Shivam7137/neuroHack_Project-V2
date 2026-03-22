from __future__ import annotations

import numpy as np

from src.preprocessing.cleanup import apply_cleanup_treatment


def test_cleanup_none_is_identity() -> None:
    signal = np.vstack([np.linspace(-1.0, 1.0, 128), np.sin(np.linspace(0.0, 8.0, 128))])
    cleaned = apply_cleanup_treatment(signal, sampling_rate=128.0, cleanup_level="none")
    assert np.allclose(cleaned, signal)


def test_cleanup_levels_preserve_shape_and_change_noisy_signal() -> None:
    time = np.linspace(0.0, 2.0, 256)
    signal = np.vstack(
        [
            np.sin(2 * np.pi * 10 * time) + 0.8 * np.sign(np.sin(2 * np.pi * 2.0 * time)),
            np.sin(2 * np.pi * 12 * time) + 0.4 * np.random.default_rng(7).normal(size=time.shape[0]),
        ]
    )

    light = apply_cleanup_treatment(signal, sampling_rate=128.0, cleanup_level="light")
    medium = apply_cleanup_treatment(signal, sampling_rate=128.0, cleanup_level="medium")
    heavy = apply_cleanup_treatment(signal, sampling_rate=128.0, cleanup_level="heavy")

    assert light.shape == signal.shape
    assert medium.shape == signal.shape
    assert heavy.shape == signal.shape
    assert not np.allclose(light, signal)
    assert not np.allclose(medium, light)
    assert not np.allclose(heavy, medium)
