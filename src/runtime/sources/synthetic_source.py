"""Synthetic source backed by the procedural or learned sampler."""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np

from src.config import get_settings
from src.generator.inference.sampler import GeneratorCondition, SyntheticSampler
from src.runtime.constants import CYTON_SAMPLE_RATE
from src.runtime.contracts import EEGChunk
from src.runtime.sources.base_source import EEGSource


@dataclass(slots=True)
class SyntheticConfig:
    """Runtime configuration for synthetic EEG generation."""

    concentration: float = 0.5
    stress: float = 0.2
    seed: int = 42
    chunk_size: int = 0


class SyntheticSource(EEGSource):
    """Canonical synthetic source for runtime development and demos."""

    source_name = "synthetic"

    def __init__(
        self,
        sampler: SyntheticSampler | None = None,
        condition: GeneratorCondition | None = None,
        chunk_seconds: float | None = None,
        config: SyntheticConfig | None = None,
        channel_names: list[str] | None = None,
    ) -> None:
        settings = get_settings()
        resolved_chunk_seconds = float(chunk_seconds or settings.runtime_chunk_seconds)
        default_config = SyntheticConfig(
            concentration=float(condition.concentration_level) if condition is not None else 0.5,
            stress=float(condition.stress_level) if condition is not None else 0.2,
            seed=settings.random_seed,
            chunk_size=max(1, int(round(resolved_chunk_seconds * CYTON_SAMPLE_RATE))),
        )
        self.config = config or default_config
        if self.config.chunk_size <= 0:
            self.config.chunk_size = max(1, int(round(resolved_chunk_seconds * CYTON_SAMPLE_RATE)))
        self.chunk_seconds = float(self.config.chunk_size / CYTON_SAMPLE_RATE)
        self.sampler = sampler or SyntheticSampler(
            channel_names=list(channel_names or settings.cyton_channel_names),
            sample_rate=float(CYTON_SAMPLE_RATE),
            random_seed=self.config.seed,
        )
        self.condition = condition or GeneratorCondition(
            concentration_level=float(self.config.concentration),
            stress_level=float(self.config.stress),
        )
        self.carry_state: dict[str, object] | None = None
        self.next_timestamp = 0.0
        self.started = False

    def set_condition(self, concentration: float, stress: float) -> None:
        self.config.concentration = float(concentration)
        self.config.stress = float(stress)
        self.condition = GeneratorCondition(
            concentration_level=float(concentration),
            stress_level=float(stress),
        )

    def set_seed(self, seed: int) -> None:
        self.config.seed = int(seed)
        if hasattr(self.sampler, "rng"):
            self.sampler.rng = np.random.default_rng(self.config.seed)
        self.carry_state = None

    def start(self) -> None:
        self.carry_state = None
        self.next_timestamp = time.time()
        self.started = True

    def stop(self) -> None:
        self.started = False

    def read_chunk(self) -> EEGChunk:
        if not self.started:
            raise RuntimeError("SyntheticSource.start() must be called before read_chunk().")
        sample = self.sampler.sample(self.condition, self.chunk_seconds, carry_state=self.carry_state)
        chunk = EEGChunk(
            timestamp_start=self.next_timestamp,
            timestamp_end=self.next_timestamp + self.chunk_seconds,
            sample_rate=float(sample.sample_rate),
            channel_names=list(sample.channel_names),
            data=sample.data,
            metadata=dict(sample.metadata),
        )
        self.carry_state = dict(sample.carry_state)
        self.next_timestamp += self.chunk_seconds
        return chunk
