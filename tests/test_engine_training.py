"""Tests for teacher-guided procedural engine fitting."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.generator.inference.sampler import GeneratorCondition, ProceduralEngineConfig, SyntheticSampler
from src.generator.training.train_engine_from_teacher import fit_procedural_engine, save_fit_result


class _FakeTeacher:
    class _FakeScorer:
        def __init__(self) -> None:
            self.models = {
                "concentration": type("ModelWrap", (), {"preprocessor": type("Pre", (), {"channel_names": ["C3", "C4", "P3", "P4"]})()})(),
                "stress": type("ModelWrap", (), {"preprocessor": type("Pre", (), {"channel_names": ["AF3", "F3", "F4", "AF4"]})()})(),
            }

    def __init__(self) -> None:
        self.scorer = self._FakeScorer()

    def predict_concentration(self, window: np.ndarray, sampling_rate: float, channel_names: list[str] | None = None) -> float:
        _ = sampling_rate, channel_names
        return float(np.clip(np.mean(np.abs(window)) / 3.0, 0.0, 1.0))

    def predict_stress(self, window: np.ndarray, sampling_rate: float, channel_names: list[str] | None = None) -> float:
        _ = sampling_rate, channel_names
        return float(np.clip(np.std(window) / 3.0, 0.0, 1.0))


def test_procedural_engine_config_round_trip() -> None:
    config = ProceduralEngineConfig()
    rebuilt = ProceduralEngineConfig.from_vector(config.to_vector())
    assert rebuilt.to_dict() == config.to_dict()


def test_synthetic_sampler_uses_engine_config() -> None:
    config = ProceduralEngineConfig(alpha_base=1.5, beta_base=0.1, theta_base=0.1, noise_base=0.001)
    sampler = SyntheticSampler(
        channel_names=["C3", "C4", "P3", "P4"],
        sample_rate=250.0,
        random_seed=3,
        engine_config=config,
    )
    sample = sampler.sample(GeneratorCondition(concentration_level=0.8, stress_level=0.1), duration_sec=2.0)
    assert sample.data.shape == (4, 500)
    assert np.isfinite(sample.data).all()
    assert sample.metadata["alpha_base"] == 1.5


def test_fit_procedural_engine_with_fake_teacher(tmp_path: Path) -> None:
    teacher = _FakeTeacher()
    result = fit_procedural_engine(teacher, task_name="concentration", maxiter=4)
    assert result.task_name == "concentration"
    assert np.isfinite(result.objective)
    saved = save_fit_result(result, tmp_path)
    assert saved.exists()
