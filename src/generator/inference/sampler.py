"""Synthetic EEG sampler with procedural and future learned backends."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from src.config import get_settings


@dataclass(slots=True)
class GeneratorCondition:
    """Public conditioning interface for synthetic EEG generation."""

    concentration_level: float
    stress_level: float
    session_style: int | None = None
    subject_style: int | None = None


@dataclass(slots=True)
class SyntheticSample:
    """Sampled canonical EEG plus carry state for continuity."""

    data: np.ndarray
    sample_rate: float
    channel_names: list[str]
    carry_state: dict[str, np.ndarray] = field(default_factory=dict)
    metadata: dict[str, float | int | str | None] = field(default_factory=dict)


@dataclass(slots=True)
class ProceduralEngineConfig:
    """Serializable parameters for the procedural EEG engine."""

    alpha_base: float = 0.9
    alpha_concentration: float = 0.8
    alpha_stress: float = -0.3
    beta_base: float = 0.3
    beta_concentration: float = 1.2
    beta_stress: float = 0.8
    theta_base: float = 0.4
    theta_concentration: float = -0.2
    theta_stress: float = 0.6
    drift_base: float = 0.05
    drift_stress: float = 0.15
    noise_base: float = 0.04
    noise_stress: float = 0.03
    channel_offset_step: float = 0.01

    @classmethod
    def from_vector(cls, vector: np.ndarray) -> "ProceduralEngineConfig":
        keys = list(asdict(cls()).keys())
        return cls(**{key: float(value) for key, value in zip(keys, vector, strict=True)})

    @classmethod
    def from_dict(cls, payload: dict[str, float]) -> "ProceduralEngineConfig":
        return cls(**{key: float(value) for key, value in payload.items()})

    def to_vector(self) -> np.ndarray:
        return np.asarray([float(value) for value in asdict(self).values()], dtype=float)

    def to_dict(self) -> dict[str, float]:
        return {key: float(value) for key, value in asdict(self).items()}


@lru_cache(maxsize=1)
def _load_teacher_guided_profiles() -> dict[str, ProceduralEngineConfig]:
    """Load saved procedural engine fits when they are available."""
    root = Path(__file__).resolve().parents[3]
    profiles: dict[str, ProceduralEngineConfig] = {}
    for task_name in ("concentration", "stress"):
        path = root / "artifacts" / "generator" / f"{task_name}_procedural_engine.json"
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            engine_payload = payload.get("engine_config", {})
            if isinstance(engine_payload, dict):
                profiles[task_name] = ProceduralEngineConfig.from_dict(engine_payload)
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            continue
    return profiles


class SyntheticSampler:
    """Procedural sampler now, learned backend hook later."""

    def __init__(
        self,
        channel_names: list[str] | None = None,
        sample_rate: float = 250.0,
        generator_backend: object | None = None,
        random_seed: int | None = None,
        engine_config: ProceduralEngineConfig | None = None,
    ) -> None:
        settings = get_settings()
        self.channel_names = list(channel_names or settings.cyton_channel_names)
        self.sample_rate = float(sample_rate)
        self.generator_backend = generator_backend
        self.rng = np.random.default_rng(random_seed if random_seed is not None else settings.random_seed)
        self.engine_config = engine_config
        self.teacher_profiles = _load_teacher_guided_profiles() if engine_config is None else {}

    def sample(
        self,
        condition: GeneratorCondition,
        duration_sec: float,
        carry_state: dict[str, np.ndarray] | None = None,
    ) -> SyntheticSample:
        if self.generator_backend is not None and hasattr(self.generator_backend, "sample"):
            return self.generator_backend.sample(condition, duration_sec, carry_state=carry_state)
        n_samples = int(round(duration_sec * self.sample_rate))
        time = np.arange(n_samples, dtype=float) / self.sample_rate
        previous_phases = None if carry_state is None else carry_state.get("phases")
        phases = previous_phases.copy() if previous_phases is not None else self.rng.uniform(
            0.0,
            2.0 * np.pi,
            size=(len(self.channel_names), 3),
        )
        previous_envelopes = None if carry_state is None else carry_state.get("envelope_phases")
        envelope_phases = previous_envelopes.copy() if previous_envelopes is not None else self.rng.uniform(
            0.0,
            2.0 * np.pi,
            size=(len(self.channel_names), 2),
        )
        previous_frequency_offsets = None if carry_state is None else carry_state.get("frequency_offsets")
        frequency_offsets = previous_frequency_offsets.copy() if previous_frequency_offsets is not None else self.rng.normal(
            loc=0.0,
            scale=np.array([0.45, 0.7, 0.3], dtype=float),
            size=(len(self.channel_names), 3),
        )
        previous_channel_gains = None if carry_state is None else carry_state.get("channel_gains")
        channel_gains = previous_channel_gains.copy() if previous_channel_gains is not None else self.rng.uniform(
            0.9,
            1.15,
            size=(len(self.channel_names), 3),
        )
        concentration = float(np.clip(condition.concentration_level, 0.0, 1.0))
        stress = float(np.clip(condition.stress_level, 0.0, 1.0))
        engine_config = self._resolve_engine_config(concentration, stress)
        alpha_amp = (
            engine_config.alpha_base
            + engine_config.alpha_concentration * concentration
            + engine_config.alpha_stress * stress
        )
        beta_amp = (
            engine_config.beta_base
            + engine_config.beta_concentration * concentration
            + engine_config.beta_stress * stress
        )
        theta_amp = (
            engine_config.theta_base
            + engine_config.theta_concentration * concentration
            + engine_config.theta_stress * stress
        )
        slow_drift_amp = engine_config.drift_base + engine_config.drift_stress * stress
        common_noise = self._smooth_noise(
            n_samples,
            scale=0.015 + 0.025 * stress,
            kernel_size=max(7, int(round(self.sample_rate * 0.06))),
        )
        common_drift = self._smooth_noise(
            n_samples,
            scale=0.04 + 0.05 * stress,
            kernel_size=max(11, int(round(self.sample_rate * 0.22))),
        )
        rows: list[np.ndarray] = []
        for channel_index, channel_name in enumerate(self.channel_names):
            channel_phase = phases[channel_index]
            region_weights = self._region_weights(channel_name)
            channel_gain = channel_gains[channel_index]
            alpha_frequency = 9.2 + 1.8 * concentration - 0.5 * stress + frequency_offsets[channel_index, 0]
            beta_frequency = 17.0 + 5.0 * concentration + 1.1 * stress + frequency_offsets[channel_index, 1]
            theta_frequency = 5.0 + 1.0 * stress - 0.3 * concentration + frequency_offsets[channel_index, 2]
            alpha_envelope = self._build_envelope(
                time,
                envelope_phases[channel_index, 0],
                base_strength=1.0 + 0.25 * concentration - 0.15 * stress,
                scale=0.16 + 0.06 * stress,
                slow_noise_scale=0.03 + 0.02 * stress,
            )
            beta_envelope = self._build_envelope(
                time,
                envelope_phases[channel_index, 1],
                base_strength=0.9 + 0.15 * concentration + 0.08 * stress,
                scale=0.12 + 0.05 * stress,
                slow_noise_scale=0.025 + 0.015 * concentration,
            )
            theta_envelope = 1.0 + 0.18 * np.sin(2.0 * np.pi * (0.11 + 0.03 * stress) * time + channel_phase[2] * 0.4)

            alpha = (
                alpha_amp
                * region_weights[0]
                * channel_gain[0]
                * alpha_envelope
                * np.sin(2.0 * np.pi * alpha_frequency * time + channel_phase[0])
            )
            beta = (
                beta_amp
                * region_weights[1]
                * channel_gain[1]
                * beta_envelope
                * np.sin(2.0 * np.pi * beta_frequency * time + channel_phase[1])
            )
            theta = (
                theta_amp
                * region_weights[2]
                * channel_gain[2]
                * theta_envelope
                * np.sin(2.0 * np.pi * theta_frequency * time + channel_phase[2])
            )
            drift = slow_drift_amp * region_weights[3] * np.sin(
                2.0 * np.pi * (0.35 + 0.08 * stress) * time + channel_index * 0.31
            )
            colored_noise = self._smooth_noise(
                n_samples,
                scale=max(0.012 + 0.01 * stress, 1e-4),
                kernel_size=max(5, int(round(self.sample_rate * 0.035))),
            )
            high_noise = self.rng.normal(
                loc=0.0,
                scale=max(engine_config.noise_base + engine_config.noise_stress * stress, 1e-4),
                size=n_samples,
            )
            signal = (
                alpha
                + beta
                + theta
                + drift
                + 0.35 * common_drift
                + 0.45 * common_noise
                + 0.55 * colored_noise
                + 0.25 * high_noise
                + channel_index * engine_config.channel_offset_step
            )
            signal += self._blink_artifact(channel_name, n_samples, concentration, stress)
            rows.append(signal)

            phase_advances = np.array([alpha_frequency, beta_frequency, theta_frequency], dtype=float)
            phases[channel_index] = (channel_phase + phase_advances * (2.0 * np.pi * duration_sec)) % (2.0 * np.pi)
            envelope_phases[channel_index] = (
                envelope_phases[channel_index]
                + np.array([0.17 + 0.03 * concentration, 0.23 + 0.05 * stress], dtype=float) * (2.0 * np.pi * duration_sec)
            ) % (2.0 * np.pi)
        return SyntheticSample(
            data=np.vstack(rows).astype(float),
            sample_rate=self.sample_rate,
            channel_names=list(self.channel_names),
            carry_state={
                "phases": phases,
                "envelope_phases": envelope_phases,
                "frequency_offsets": frequency_offsets,
                "channel_gains": channel_gains,
            },
            metadata={
                "concentration_level": concentration,
                "stress_level": stress,
                **engine_config.to_dict(),
            },
        )

    def _resolve_engine_config(self, concentration: float, stress: float) -> ProceduralEngineConfig:
        if self.engine_config is not None:
            return self.engine_config
        base = ProceduralEngineConfig()
        vector = base.to_vector()
        concentration_profile = self.teacher_profiles.get("concentration")
        stress_profile = self.teacher_profiles.get("stress")
        if concentration_profile is not None:
            vector += 0.2 * concentration * (concentration_profile.to_vector() - base.to_vector())
        if stress_profile is not None:
            vector += 0.2 * stress * (stress_profile.to_vector() - base.to_vector())
        return ProceduralEngineConfig.from_vector(vector)

    def _region_weights(self, channel_name: str) -> np.ndarray:
        upper = channel_name.upper()
        if upper.startswith("FP"):
            return np.array([0.72, 0.95, 1.24, 1.12], dtype=float)
        if upper.startswith("F"):
            return np.array([0.82, 1.08, 1.16, 1.02], dtype=float)
        if upper.startswith("C"):
            return np.array([0.95, 1.12, 0.96, 0.9], dtype=float)
        if upper.startswith("P"):
            return np.array([1.18, 0.88, 0.84, 0.78], dtype=float)
        if upper.startswith("O"):
            return np.array([1.34, 0.74, 0.72, 0.62], dtype=float)
        if upper.startswith("T"):
            return np.array([0.88, 0.96, 1.0, 0.86], dtype=float)
        return np.ones(4, dtype=float)

    def _build_envelope(
        self,
        time: np.ndarray,
        phase: float,
        base_strength: float,
        scale: float,
        slow_noise_scale: float,
    ) -> np.ndarray:
        envelope = base_strength + scale * np.sin(2.0 * np.pi * 0.18 * time + phase)
        envelope += 0.5 * scale * np.sin(2.0 * np.pi * 0.07 * time + phase * 0.37)
        envelope += self._smooth_noise(
            time.size,
            scale=slow_noise_scale,
            kernel_size=max(9, int(round(self.sample_rate * 0.12))),
        )
        return np.clip(envelope, 0.2, None)

    def _smooth_noise(self, length: int, scale: float, kernel_size: int) -> np.ndarray:
        white = self.rng.normal(loc=0.0, scale=scale, size=length + kernel_size)
        kernel_points = np.linspace(-2.5, 2.5, kernel_size, dtype=float)
        kernel = np.exp(-(kernel_points**2))
        kernel /= np.sum(kernel)
        filtered = np.convolve(white, kernel, mode="valid")
        return filtered[:length]

    def _blink_artifact(
        self,
        channel_name: str,
        n_samples: int,
        concentration: float,
        stress: float,
    ) -> np.ndarray:
        upper = channel_name.upper()
        if not upper.startswith("FP"):
            return np.zeros(n_samples, dtype=float)
        probability = 0.05 + 0.12 * stress + 0.05 * (1.0 - concentration)
        if self.rng.random() > probability:
            return np.zeros(n_samples, dtype=float)
        center = int(self.rng.integers(max(1, n_samples // 8), max(2, n_samples - n_samples // 8)))
        width = max(6.0, 0.025 * self.sample_rate)
        samples = np.arange(n_samples, dtype=float)
        pulse = np.exp(-0.5 * ((samples - center) / width) ** 2)
        return (0.18 + 0.24 * stress) * pulse
