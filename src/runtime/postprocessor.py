"""Runtime decision smoothing and state stabilization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.runtime.adaptation import AdaptationProfile
from src.runtime.contracts import DecisionOutput, DecisionScores


@dataclass(slots=True)
class PostprocessorConfig:
    """Default runtime decision stabilization settings."""

    smoothing_alpha: float = 0.35
    concentration_threshold: float = 0.6
    stress_threshold: float = 0.65
    hysteresis_margin: float = 0.05


class DecisionPostprocessor:
    """Convert raw model scores into stable runtime states."""

    def __init__(self, config: PostprocessorConfig | None = None) -> None:
        self.config = config or PostprocessorConfig()
        self._concentration_smoothed: float | None = None
        self._stress_smoothed: float | None = None
        self._last_state = "neutral"

    def reset(self) -> None:
        self._concentration_smoothed = None
        self._stress_smoothed = None
        self._last_state = "neutral"

    def update(
        self,
        scores: DecisionScores,
        *,
        timestamp_start: float,
        timestamp_end: float,
        profile: AdaptationProfile | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> DecisionOutput:
        alpha = self._effective_smoothing_alpha(profile)
        concentration_smoothed = self._smooth(self._concentration_smoothed, scores.concentration, alpha)
        stress_smoothed = self._smooth(self._stress_smoothed, scores.stress, alpha)
        self._concentration_smoothed = concentration_smoothed
        self._stress_smoothed = stress_smoothed

        concentration_threshold = self.config.concentration_threshold + self._profile_shift(profile, "concentration")
        stress_threshold = self.config.stress_threshold + self._profile_shift(profile, "stress")
        state = self._decide_state(
            concentration_smoothed=concentration_smoothed,
            stress_smoothed=stress_smoothed,
            concentration_threshold=concentration_threshold,
            stress_threshold=stress_threshold,
        )
        self._last_state = state
        return DecisionOutput(
            timestamp_start=float(timestamp_start),
            timestamp_end=float(timestamp_end),
            concentration_raw=float(scores.concentration),
            stress_raw=float(scores.stress),
            concentration_smoothed=float(concentration_smoothed),
            stress_smoothed=float(stress_smoothed),
            state=state,
            confidence=float(scores.confidence),
            quality=float(scores.quality),
            metadata={
                **dict(metadata or {}),
                "concentration_threshold": float(concentration_threshold),
                "stress_threshold": float(stress_threshold),
                "smoothing_alpha": float(alpha),
            },
        )

    def _effective_smoothing_alpha(self, profile: AdaptationProfile | None) -> float:
        if profile is not None and profile.enabled and profile.smoothing_alpha > 0.0:
            return float(profile.smoothing_alpha)
        return float(self.config.smoothing_alpha)

    @staticmethod
    def _smooth(previous: float | None, current: float, alpha: float) -> float:
        if previous is None:
            return float(current)
        return float((alpha * current) + ((1.0 - alpha) * previous))

    def _profile_shift(self, profile: AdaptationProfile | None, signal_name: str) -> float:
        if profile is None or not profile.enabled:
            return 0.0
        if signal_name == "concentration":
            return float(profile.concentration_threshold_shift)
        return float(profile.stress_threshold_shift)

    def _decide_state(
        self,
        *,
        concentration_smoothed: float,
        stress_smoothed: float,
        concentration_threshold: float,
        stress_threshold: float,
    ) -> str:
        focused = self._passes_threshold(
            value=concentration_smoothed,
            threshold=concentration_threshold,
            active_state="focused",
        ) and stress_smoothed < (stress_threshold - self.config.hysteresis_margin)
        stressed = self._passes_threshold(
            value=stress_smoothed,
            threshold=stress_threshold,
            active_state="stressed",
        )
        if focused and stressed:
            return "mixed"
        if stressed:
            return "stressed"
        if focused:
            return "focused"
        return "neutral"

    def _passes_threshold(self, *, value: float, threshold: float, active_state: str) -> bool:
        margin = self.config.hysteresis_margin
        if self._last_state == active_state:
            return bool(value >= (threshold - margin))
        return bool(value >= threshold)
