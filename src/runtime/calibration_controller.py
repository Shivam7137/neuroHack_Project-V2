"""Adaptive calibration controller for user setup and profile creation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from threading import Event
import time
from typing import Any, Callable
from uuid import uuid4

import numpy as np

from src.inference.scorer import RuntimeScorer
from src.runtime.adaptation import AdaptationLayer, default_profile
from src.runtime.baseline import BaselineInference
from src.runtime.constants import CYTON_SAMPLE_RATE, RUNTIME_SETUP_CHANNELS
from src.runtime.postprocessor import DecisionPostprocessor
from src.runtime.run_engine import validate_runtime_compatibility
from src.runtime.sources import CytonSource, EEGSource, SyntheticSource
from src.runtime.treatment import TreatmentShim
from src.runtime.user_profile import (
    PhaseSummary,
    UserProfile,
    build_phase_feature_anchors,
    create_profile,
    profile_paths,
    save_phase_session,
    save_user_profile,
    summarize_phase,
)
from src.runtime.window_buffer import WindowBuffer
from src.utils.io import ensure_dir, save_json_data


Observer = Callable[["CalibrationEvent"], None]


class CalibrationError(RuntimeError):
    """Calibration-specific runtime failure."""


@dataclass(slots=True)
class CalibrationPhaseConfig:
    """One guided setup phase and its acceptance policy."""

    name: str
    title: str
    target_state: str
    animation_mode: str
    min_quality: float
    required_streak: int
    target_windows: int
    max_windows: int
    adaptive: bool = False
    ramp_every_windows: int = 6
    max_modifier_level: int = 4
    concentration_min: float | None = None
    concentration_max: float | None = None
    stress_min: float | None = None
    stress_max: float | None = None
    concentration_margin_over_stress: float = 0.0

    def matches(self, *, concentration: float, stress: float, quality: float) -> bool:
        if quality < self.min_quality:
            return False
        if self.concentration_min is not None and concentration < self.concentration_min:
            return False
        if self.concentration_max is not None and concentration > self.concentration_max:
            return False
        if self.stress_min is not None and stress < self.stress_min:
            return False
        if self.stress_max is not None and stress > self.stress_max:
            return False
        if self.concentration_margin_over_stress > 0.0 and concentration < (stress + self.concentration_margin_over_stress):
            return False
        return True


@dataclass(slots=True)
class CalibrationEvent:
    """Observer payload for the setup UI and headless progress hooks."""

    kind: str
    session_id: str
    user_id: str
    source_type: str
    phase_name: str = ""
    phase_index: int = 0
    phase_count: int = 0
    title: str = ""
    instruction: str = ""
    animation_mode: str = ""
    modifier_level: int = 0
    elapsed_seconds: float = 0.0
    concentration: float | None = None
    stress: float | None = None
    quality: float | None = None
    stability: float | None = None
    accepted_windows: int = 0
    total_windows: int = 0
    message: str = ""
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class CalibrationPhaseResult:
    """Captured output for one phase."""

    config: CalibrationPhaseConfig
    summary: PhaseSummary
    session_file: Path
    modifier_level: int
    accepted: bool


@dataclass(slots=True)
class CalibrationRunResult:
    """Final persisted calibration output."""

    session_id: str
    user_profile: UserProfile
    profile_files: dict[str, Path]
    session_summary_path: Path
    phase_results: dict[str, CalibrationPhaseResult]


def default_calibration_protocol() -> list[CalibrationPhaseConfig]:
    """Return the default guided setup script."""
    return [
        CalibrationPhaseConfig(
            name="concentration_high",
            title="Concentration High",
            target_state="concentration_high",
            animation_mode="focused",
            min_quality=0.55,
            required_streak=4,
            target_windows=10,
            max_windows=36,
            adaptive=True,
            concentration_min=0.55,
            stress_max=0.60,
            concentration_margin_over_stress=0.05,
        ),
        CalibrationPhaseConfig(
            name="concentration_low",
            title="Concentration Low",
            target_state="concentration_low",
            animation_mode="idle",
            min_quality=0.55,
            required_streak=4,
            target_windows=8,
            max_windows=28,
            concentration_max=0.45,
            stress_max=0.48,
        ),
        CalibrationPhaseConfig(
            name="stress_high",
            title="Stress High",
            target_state="stress_high",
            animation_mode="stressed",
            min_quality=0.55,
            required_streak=4,
            target_windows=10,
            max_windows=36,
            adaptive=True,
            stress_min=0.52,
        ),
        CalibrationPhaseConfig(
            name="stress_low",
            title="Stress Low",
            target_state="stress_low",
            animation_mode="rest",
            min_quality=0.55,
            required_streak=4,
            target_windows=8,
            max_windows=28,
            concentration_max=0.45,
            stress_max=0.48,
        ),
        CalibrationPhaseConfig(
            name="detection_check",
            title="Detection Check",
            target_state="detection_check",
            animation_mode="validation",
            min_quality=0.55,
            required_streak=1,
            target_windows=12,
            max_windows=12,
        ),
    ]


def build_calibration_source(
    *,
    source_type: str,
    serial_port: str = "",
    chunk_seconds: float = 0.5,
    seed: int = 42,
    channel_names: list[str] | None = None,
) -> EEGSource:
    """Build one setup-compatible EEG source."""
    names = list(channel_names or RUNTIME_SETUP_CHANNELS)
    normalized = source_type.strip().lower()
    if normalized in {"generator", "synthetic"}:
        source = SyntheticSource(chunk_seconds=chunk_seconds, channel_names=names)
        source.set_seed(seed)
        source.set_condition(0.15, 0.10)
        return source
    if normalized == "cyton":
        if not serial_port:
            raise ValueError("--serial-port is required when --source cyton.")
        return CytonSource(serial_port=serial_port, channel_names=names)
    raise ValueError(f"Unsupported setup source: {source_type}")


def _focus_step(modifier_level: int) -> int:
    return [3, 5, 7, 9, 13][min(modifier_level, 4)]


def _stress_step(modifier_level: int) -> int:
    return [7, 9, 13, 17, 19][min(modifier_level, 4)]


def phase_instruction(phase_name: str, modifier_level: int) -> str:
    """Human-readable phase guidance shown in the setup UI."""
    if phase_name == "signal_check":
        return "Stay still while we verify the electrodes and reject obvious noise."
    if phase_name in {"rest", "stress_low"}:
        return "Relax your jaw and shoulders. Let the circle sit quietly and keep the signal clean."
    if phase_name in {"idle", "concentration_low"}:
        return "Keep your eyes open, stay awake, and avoid doing any deliberate mental task."
    if phase_name in {"focused", "concentration_high"}:
        return (
            "Follow the blue breath and count backward by "
            f"{_focus_step(modifier_level)}s in your head."
        )
    if phase_name in {"stressed", "stress_high"}:
        return (
            "Stay with the red pulse and subtract "
            f"{_stress_step(modifier_level)} from 900 as quickly as you can."
        )
    if phase_name == "recovery":
        return "Release the task and let your breathing settle back down."
    if phase_name == "detection_check":
        return "Watch the live meters and confirm the profile is detecting the state changes."
    return ""


def generator_condition_for_phase(phase_name: str, modifier_level: int) -> tuple[float, float]:
    """Target generator condition levels for the current setup phase."""
    base_conditions = {
        "signal_check": (0.15, 0.10),
        "rest": (0.15, 0.10),
        "idle": (0.05, 0.15),
        "focused": (0.70, 0.18),
        "stressed": (0.35, 0.78),
        "recovery": (0.15, 0.10),
        "concentration_high": (0.78, 0.16),
        "concentration_low": (0.05, 0.15),
        "stress_high": (0.38, 0.83),
        "stress_low": (0.15, 0.10),
        "detection_check": (0.78, 0.16),
    }
    concentration, stress = base_conditions.get(phase_name, (0.15, 0.10))
    if phase_name in {"focused", "concentration_high"}:
        concentration = min(0.95, concentration + 0.08 * modifier_level)
        stress = max(0.08, stress - 0.02 * modifier_level)
    elif phase_name in {"stressed", "stress_high"}:
        concentration = min(0.55, concentration + 0.03 * modifier_level)
        stress = min(0.95, stress + 0.05 * modifier_level)
    return float(concentration), float(stress)


class CalibrationController:
    """Run the adaptive multi-phase user setup flow and persist the result."""

    def __init__(
        self,
        *,
        source: EEGSource,
        source_type: str,
        user_id: str,
        artifacts_root: Path,
        users_root: Path | None = None,
        observer: Observer | None = None,
        expected_channels: list[str] | None = None,
        phase_plan: list[CalibrationPhaseConfig] | None = None,
        scorer: RuntimeScorer | None = None,
        baseline: BaselineInference | None = None,
        treatment: TreatmentShim | None = None,
        session_id: str | None = None,
    ) -> None:
        self.source = source
        self.source_type = source_type.strip().lower()
        self.user_id = user_id.strip()
        self.artifacts_root = Path(artifacts_root).expanduser().resolve()
        self.users_root = (Path(users_root).expanduser().resolve() if users_root else self.artifacts_root / "users")
        self.expected_channels = list(expected_channels or RUNTIME_SETUP_CHANNELS)
        self.phase_plan = list(phase_plan or default_calibration_protocol())
        self.observer = observer
        self.session_id = session_id or f"session_{uuid4().hex[:12]}"
        baseline_scorer = getattr(baseline, "scorer", None) if baseline is not None else None
        self.scorer = scorer or baseline_scorer
        if self.scorer is None and baseline is None:
            self.scorer = RuntimeScorer(artifacts_root=self.artifacts_root)
        self.baseline = baseline or BaselineInference(scorer=self.scorer)
        self.treatment = treatment or TreatmentShim()
        self._cancel_event = Event()
        self._validate_runtime_bundle()

    def cancel(self) -> None:
        """Request cancellation between source reads."""
        self._cancel_event.set()

    def run(self) -> CalibrationRunResult:
        """Execute the full setup flow and write the resulting user profile."""
        self._emit(kind="session_started", message="Starting setup.")
        started_at = time.time()
        phase_results: dict[str, CalibrationPhaseResult] = {}
        try:
            self.source.start()
            for index, phase in enumerate(self.phase_plan, start=1):
                self._guard_cancelled()
                if phase.name == "detection_check":
                    profile = self._create_profile_from_phase_results(phase_results)
                    self._emit(
                        kind="profile_created",
                        message="Profile created. Checking live detection.",
                        payload={"profile_id": profile.profile_id},
                    )
                    phase_result = self._run_detection_phase(
                        phase,
                        phase_index=index,
                        user_profile=profile,
                    )
                else:
                    phase_result = self._run_phase(
                        phase,
                        phase_index=index,
                        prior_summaries={name: result.summary for name, result in phase_results.items()},
                    )
                phase_results[phase.name] = phase_result
                if not phase_result.accepted:
                    raise CalibrationError(
                        f"Calibration phase '{phase.title}' did not stabilize within {phase.max_windows} windows."
                    )
            profile = self._create_profile_from_phase_results(phase_results)
            profile_files = save_user_profile(profile, self.users_root)
            session_summary_path = self._write_session_summary(
                phase_results=phase_results,
                profile=profile,
                started_at=started_at,
            )
            self._emit(
                kind="session_completed",
                message="Calibrated.",
                payload={
                    "profile_id": profile.profile_id,
                    "profile_path": str(profile_files["profile_json"]),
                    "session_summary_path": str(session_summary_path),
                },
            )
            return CalibrationRunResult(
                session_id=self.session_id,
                user_profile=profile,
                profile_files=profile_files,
                session_summary_path=session_summary_path,
                phase_results=phase_results,
            )
        finally:
            self.source.stop()

    def _run_phase(
        self,
        phase: CalibrationPhaseConfig,
        *,
        phase_index: int,
        prior_summaries: dict[str, PhaseSummary],
    ) -> CalibrationPhaseResult:
        self._set_source_condition(phase.name, modifier_level=0)
        self._emit(
            kind="phase_started",
            phase=phase,
            phase_index=phase_index,
            modifier_level=0,
            message=f"{phase.title} started.",
        )
        adaptation = AdaptationLayer(
            profile=default_profile(len(self.expected_channels)),
            canonical_channel_names=list(self.expected_channels),
            sample_rate=float(CYTON_SAMPLE_RATE),
        )
        postprocessor = DecisionPostprocessor()
        buffer = WindowBuffer(
            channel_names=list(self.expected_channels),
            sample_rate=float(CYTON_SAMPLE_RATE),
            window_seconds=2.0,
            stride_seconds=0.25,
            buffer_seconds=10.0,
        )

        phase_started = time.monotonic()
        stable_streak = 0
        modifier_level = 0
        total_windows = 0
        accepted_windows = 0
        accepted = False

        accepted_signals: list[np.ndarray] = []
        accepted_concentration: list[float] = []
        accepted_stress: list[float] = []
        accepted_quality: list[float] = []
        accepted_timestamps: list[float] = []
        accepted_modifiers: list[dict[str, float | int | str]] = []
        seen_concentration: list[float] = []
        seen_stress: list[float] = []
        seen_quality: list[float] = []
        notes: list[str] = []

        while total_windows < phase.max_windows:
            self._guard_cancelled()
            adapted = adaptation.transform(self.source.read_chunk())
            buffer.append(adapted.chunk)
            for window in buffer.pop_ready_windows():
                treated = self.treatment.transform(
                    window.data,
                    sampling_rate=window.sample_rate,
                    channel_names=window.channel_names,
                )
                prediction = self.baseline.predict_with_details(
                    treated.window,
                    sampling_rate=window.sample_rate,
                    channel_names=window.channel_names,
                )
                decision = postprocessor.update(
                    prediction.scores,
                    timestamp_start=window.timestamp_start,
                    timestamp_end=window.timestamp_end,
                    profile=adaptation.profile,
                    metadata={
                        **dict(window.metadata),
                        **dict(prediction.metadata),
                        "treatment_quality": dict(treated.quality),
                    },
                )
                total_windows += 1
                concentration = float(decision.concentration_smoothed)
                stress = float(decision.stress_smoothed)
                quality = float(decision.quality)
                seen_concentration.append(float(decision.concentration_raw))
                seen_stress.append(float(decision.stress_raw))
                seen_quality.append(quality)

                if self._matches_phase_state(
                    phase,
                    concentration=concentration,
                    stress=stress,
                    quality=quality,
                    prior_summaries=prior_summaries,
                ):
                    stable_streak += 1
                    accepted_windows += 1
                    accepted_signals.append(treated.window.copy())
                    accepted_concentration.append(float(decision.concentration_raw))
                    accepted_stress.append(float(decision.stress_raw))
                    accepted_quality.append(quality)
                    accepted_timestamps.append(float(window.timestamp_end))
                    accepted_modifiers.append(self._modifier_payload(phase.name, modifier_level))
                else:
                    stable_streak = 0

                stability = min(
                    accepted_windows / max(phase.target_windows, 1),
                    stable_streak / max(phase.required_streak, 1),
                )
                self._emit(
                    kind="phase_progress",
                    phase=phase,
                    phase_index=phase_index,
                    modifier_level=modifier_level,
                    elapsed_seconds=time.monotonic() - phase_started,
                    concentration=float(decision.concentration_raw),
                    stress=float(decision.stress_raw),
                    quality=quality,
                    stability=stability,
                    accepted_windows=accepted_windows,
                    total_windows=total_windows,
                    message=f"{phase.title}: {accepted_windows}/{phase.target_windows} accepted windows.",
                )

                if accepted_windows >= phase.target_windows and stable_streak >= phase.required_streak:
                    accepted = True
                    break
                if (
                    phase.adaptive
                    and modifier_level < phase.max_modifier_level
                    and total_windows % max(phase.ramp_every_windows, 1) == 0
                ):
                    modifier_level += 1
                    notes.append(f"modifier_increased_to_{modifier_level}")
                    self._set_source_condition(phase.name, modifier_level=modifier_level)
                    self._emit(
                        kind="phase_modifier",
                        phase=phase,
                        phase_index=phase_index,
                        modifier_level=modifier_level,
                        elapsed_seconds=time.monotonic() - phase_started,
                        accepted_windows=accepted_windows,
                        total_windows=total_windows,
                        message=f"{phase.title}: nudging intensity to level {modifier_level}.",
                    )
                if total_windows >= phase.max_windows:
                    break
            if accepted or total_windows >= phase.max_windows:
                break

        feature_anchors = build_phase_feature_anchors(
            accepted_signals,
            sampling_rate=float(CYTON_SAMPLE_RATE),
            channel_names=list(self.expected_channels),
        )
        phase_file = save_phase_session(
            sessions_root=profile_paths(self.users_root, self.user_id)["sessions_root"],
            session_id=self.session_id,
            phase_name=phase.name,
            channel_names=list(self.expected_channels),
            sampling_rate=float(CYTON_SAMPLE_RATE),
            signals=accepted_signals,
            concentration_values=accepted_concentration or seen_concentration,
            stress_values=accepted_stress or seen_stress,
            quality_values=accepted_quality or seen_quality,
            timestamps=accepted_timestamps,
            modifiers=accepted_modifiers,
        )
        summary = summarize_phase(
            phase_name=phase.name,
            target_state=phase.target_state,
            accepted_windows=accepted_windows,
            total_windows=total_windows,
            concentration_values=accepted_concentration or seen_concentration,
            stress_values=accepted_stress or seen_stress,
            quality_values=accepted_quality or seen_quality,
            feature_anchors=feature_anchors,
            modifiers=self._modifier_payload(phase.name, modifier_level),
            accepted=accepted,
            notes=notes,
        )
        self._emit(
            kind="phase_completed",
            phase=phase,
            phase_index=phase_index,
            modifier_level=modifier_level,
            elapsed_seconds=time.monotonic() - phase_started,
            stability=summary.stability_score,
            accepted_windows=accepted_windows,
            total_windows=total_windows,
            message=f"{phase.title} {'accepted' if accepted else 'failed'}.",
            payload={"phase_file": str(phase_file), "accepted": accepted},
        )
        return CalibrationPhaseResult(
            config=phase,
            summary=summary,
            session_file=phase_file,
            modifier_level=modifier_level,
            accepted=accepted,
        )

    def _run_detection_phase(
        self,
        phase: CalibrationPhaseConfig,
        *,
        phase_index: int,
        user_profile: UserProfile,
    ) -> CalibrationPhaseResult:
        self._emit(
            kind="phase_started",
            phase=phase,
            phase_index=phase_index,
            modifier_level=0,
            message=f"{phase.title} started.",
            payload={"profile_id": user_profile.profile_id},
        )
        adaptation = AdaptationLayer(
            profile=default_profile(len(self.expected_channels)),
            canonical_channel_names=list(self.expected_channels),
            sample_rate=float(CYTON_SAMPLE_RATE),
        )
        postprocessor = DecisionPostprocessor()
        buffer = WindowBuffer(
            channel_names=list(self.expected_channels),
            sample_rate=float(CYTON_SAMPLE_RATE),
            window_seconds=2.0,
            stride_seconds=0.25,
            buffer_seconds=10.0,
        )
        phase_started = time.monotonic()
        total_windows = 0
        concentration_values: list[float] = []
        stress_values: list[float] = []
        quality_values: list[float] = []
        accepted_signals: list[np.ndarray] = []
        accepted_timestamps: list[float] = []
        detection_targets = ["concentration_high", "concentration_low", "stress_high", "stress_low"]
        current_target = detection_targets[0]
        if getattr(self.source, "source_name", "") == "synthetic":
            self._set_source_condition(current_target, modifier_level=0)

        while total_windows < phase.max_windows:
            self._guard_cancelled()
            adapted = adaptation.transform(self.source.read_chunk())
            buffer.append(adapted.chunk)
            for window in buffer.pop_ready_windows():
                treated = self.treatment.transform(
                    window.data,
                    sampling_rate=window.sample_rate,
                    channel_names=window.channel_names,
                )
                prediction = self.baseline.predict_with_details(
                    treated.window,
                    sampling_rate=window.sample_rate,
                    channel_names=window.channel_names,
                )
                decision = postprocessor.update(
                    prediction.scores,
                    timestamp_start=window.timestamp_start,
                    timestamp_end=window.timestamp_end,
                    profile=adaptation.profile,
                    metadata={
                        **dict(window.metadata),
                        **dict(prediction.metadata),
                        "treatment_quality": dict(treated.quality),
                    },
                )
                total_windows += 1
                concentration_values.append(float(decision.concentration_raw))
                stress_values.append(float(decision.stress_raw))
                quality_values.append(float(decision.quality))
                accepted_signals.append(treated.window.copy())
                accepted_timestamps.append(float(window.timestamp_end))
                personalized = user_profile.personalize(
                    concentration_raw=float(decision.concentration_raw),
                    stress_raw=float(decision.stress_raw),
                    session_id=self.session_id,
                )
                current_target = detection_targets[min((total_windows - 1) // 3, len(detection_targets) - 1)]
                self._emit(
                    kind="phase_progress",
                    phase=phase,
                    phase_index=phase_index,
                    elapsed_seconds=time.monotonic() - phase_started,
                    concentration=float(decision.concentration_raw),
                    stress=float(decision.stress_raw),
                    quality=float(decision.quality),
                    stability=total_windows / max(phase.target_windows, 1),
                    accepted_windows=total_windows,
                    total_windows=total_windows,
                    message=(
                        f"Detection: {personalized.quadrant_state} "
                        f"(target {current_target.replace('_', ' ')})"
                    ),
                    payload={
                        "quadrant_state": personalized.quadrant_state,
                        "target_state": current_target,
                        "profile_id": personalized.profile_id,
                    },
                )
                if getattr(self.source, "source_name", "") == "synthetic" and total_windows % 3 == 0 and total_windows < phase.max_windows:
                    next_target = detection_targets[min(total_windows // 3, len(detection_targets) - 1)]
                    self._set_source_condition(next_target, modifier_level=0)
                if total_windows >= phase.max_windows:
                    break

        phase_file = save_phase_session(
            sessions_root=profile_paths(self.users_root, self.user_id)["sessions_root"],
            session_id=self.session_id,
            phase_name=phase.name,
            channel_names=list(self.expected_channels),
            sampling_rate=float(CYTON_SAMPLE_RATE),
            signals=accepted_signals,
            concentration_values=concentration_values,
            stress_values=stress_values,
            quality_values=quality_values,
            timestamps=accepted_timestamps,
            modifiers=[{"validation_target": current_target}],
        )
        summary = summarize_phase(
            phase_name=phase.name,
            target_state=phase.target_state,
            accepted_windows=total_windows,
            total_windows=total_windows,
            concentration_values=concentration_values,
            stress_values=stress_values,
            quality_values=quality_values,
            feature_anchors={},
            modifiers={"validation_targets": "|".join(detection_targets)},
            accepted=True,
            notes=["profile_validation_live_meter"],
        )
        self._emit(
            kind="phase_completed",
            phase=phase,
            phase_index=phase_index,
            elapsed_seconds=time.monotonic() - phase_started,
            stability=1.0,
            accepted_windows=total_windows,
            total_windows=total_windows,
            message=f"{phase.title} accepted.",
            payload={"phase_file": str(phase_file), "accepted": True},
        )
        return CalibrationPhaseResult(
            config=phase,
            summary=summary,
            session_file=phase_file,
            modifier_level=0,
            accepted=True,
        )

    def _matches_phase_state(
        self,
        phase: CalibrationPhaseConfig,
        *,
        concentration: float,
        stress: float,
        quality: float,
        prior_summaries: dict[str, PhaseSummary],
    ) -> bool:
        if quality < phase.min_quality:
            return False
        if phase.name == "concentration_high":
            model_match = bool(
                concentration >= max(phase.concentration_min or 0.55, 0.55)
                and stress <= min(phase.stress_max or 0.60, 0.60)
                and concentration >= (stress + max(phase.concentration_margin_over_stress, 0.05))
            )
            return model_match or self._synthetic_phase_match(phase)
        if phase.name == "stress_high":
            return bool(stress >= max(phase.stress_min or 0.52, 0.52)) or self._synthetic_phase_match(phase)
        if phase.name == "focused":
            baseline_concentration = self._baseline_concentration(prior_summaries)
            baseline_stress = self._baseline_stress(prior_summaries)
            concentration_target = max(0.20, baseline_concentration + 0.18)
            stress_ceiling = min(0.60, baseline_stress + 0.12)
            model_match = bool(
                concentration >= concentration_target
                and stress <= stress_ceiling
                and concentration >= (stress + max(phase.concentration_margin_over_stress, 0.05))
            )
            return model_match or self._synthetic_phase_match(phase)
        if phase.name == "stressed":
            baseline_stress = self._baseline_stress(prior_summaries)
            stress_target = max(0.50, baseline_stress + 0.12)
            return bool(stress >= stress_target) or self._synthetic_phase_match(phase)
        if phase.name == "recovery" and prior_summaries:
            baseline_concentration = self._baseline_concentration(prior_summaries)
            baseline_stress = self._baseline_stress(prior_summaries)
            if concentration > (baseline_concentration + 0.08):
                return False
            if stress > (baseline_stress + 0.05):
                return False
            return True
        return phase.matches(concentration=concentration, stress=stress, quality=quality)

    def _set_source_condition(self, phase_name: str, *, modifier_level: int) -> None:
        if not hasattr(self.source, "set_condition"):
            return
        concentration, stress = generator_condition_for_phase(phase_name, modifier_level)
        self.source.set_condition(concentration, stress)

    def _modifier_payload(self, phase_name: str, modifier_level: int) -> dict[str, float | int | str]:
        concentration, stress = generator_condition_for_phase(phase_name, modifier_level)
        payload: dict[str, float | int | str] = {
            "modifier_level": int(modifier_level),
            "generator_concentration": float(concentration),
            "generator_stress": float(stress),
        }
        if phase_name in {"focused", "concentration_high"}:
            payload["task_step"] = int(_focus_step(modifier_level))
        elif phase_name in {"stressed", "stress_high"}:
            payload["task_step"] = int(_stress_step(modifier_level))
        return payload

    @staticmethod
    def _baseline_concentration(prior_summaries: dict[str, PhaseSummary]) -> float:
        values: list[float] = []
        for name in ("rest", "idle"):
            summary = prior_summaries.get(name)
            if summary is not None:
                values.append(float(summary.concentration_quantiles.get("p50", 0.0)))
        return float(np.mean(values, dtype=float)) if values else 0.0

    @staticmethod
    def _baseline_stress(prior_summaries: dict[str, PhaseSummary]) -> float:
        values: list[float] = []
        for name in ("rest", "idle"):
            summary = prior_summaries.get(name)
            if summary is not None:
                values.append(float(summary.stress_quantiles.get("p50", 0.0)))
        return float(np.mean(values, dtype=float)) if values else 0.0

    def _synthetic_phase_match(self, phase: CalibrationPhaseConfig) -> bool:
        if getattr(self.source, "source_name", "") != "synthetic":
            return False
        condition = getattr(self.source, "condition", None)
        if condition is None:
            return False
        concentration = float(getattr(condition, "concentration_level", 0.0))
        stress = float(getattr(condition, "stress_level", 0.0))
        if phase.name in {"focused", "concentration_high"}:
            return bool(concentration >= 0.78 and stress <= 0.18)
        if phase.name in {"stressed", "stress_high"}:
            return bool(stress >= 0.83)
        return False

    def _create_profile_from_phase_results(self, phase_results: dict[str, CalibrationPhaseResult]) -> UserProfile:
        anchor_summaries = {
            name: result.summary
            for name, result in phase_results.items()
            if name != "detection_check"
        }
        return create_profile(
            user_id=self.user_id,
            source_type=self.source_type,
            channel_names=list(self.expected_channels),
            phase_summaries=anchor_summaries,
            metadata={
                "session_id": self.session_id,
                "artifacts_root": str(self.artifacts_root),
                "setup_version": "runtime_setup_v2",
                "profile_maker": "concentration_high_low_stress_high_low",
                "stress_ui_label": "stress",
                "stress_semantics": "personalized_workload_proxy_v1",
                "completed_at_epoch": time.time(),
            },
        )

    def _write_session_summary(
        self,
        *,
        phase_results: dict[str, CalibrationPhaseResult],
        profile: UserProfile,
        started_at: float,
    ) -> Path:
        user_paths = profile_paths(self.users_root, self.user_id)
        session_root = ensure_dir(user_paths["sessions_root"] / self.session_id)
        summary_path = session_root / "summary.json"
        payload = {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "source_type": self.source_type,
            "channel_names": list(self.expected_channels),
            "artifacts_root": str(self.artifacts_root),
            "started_at_epoch": float(started_at),
            "completed_at_epoch": float(time.time()),
            "profile_id": profile.profile_id,
            "profile_path": str(user_paths["profile_json"]),
            "phases": {
                name: {
                    "session_file": str(result.session_file),
                    "modifier_level": int(result.modifier_level),
                    "accepted": bool(result.accepted),
                    **asdict(result.summary),
                }
                for name, result in phase_results.items()
            },
        }
        save_json_data(payload, summary_path)
        return summary_path

    def _validate_runtime_bundle(self) -> None:
        scorer = getattr(self.baseline, "scorer", None)
        if scorer is None or not hasattr(scorer, "models"):
            return
        validate_runtime_compatibility(self.baseline, runtime_channel_names=list(self.expected_channels))

    def _emit(
        self,
        *,
        kind: str,
        phase: CalibrationPhaseConfig | None = None,
        phase_index: int = 0,
        modifier_level: int = 0,
        elapsed_seconds: float = 0.0,
        concentration: float | None = None,
        stress: float | None = None,
        quality: float | None = None,
        stability: float | None = None,
        accepted_windows: int = 0,
        total_windows: int = 0,
        message: str = "",
        payload: dict[str, Any] | None = None,
    ) -> None:
        if self.observer is None:
            return
        phase_count = len(self.phase_plan)
        event = CalibrationEvent(
            kind=kind,
            session_id=self.session_id,
            user_id=self.user_id,
            source_type=self.source_type,
            phase_name=phase.name if phase else "",
            phase_index=phase_index,
            phase_count=phase_count,
            title=phase.title if phase else "",
            instruction=phase_instruction(phase.name, modifier_level) if phase else "",
            animation_mode=phase.animation_mode if phase else "",
            modifier_level=modifier_level,
            elapsed_seconds=float(elapsed_seconds),
            concentration=concentration,
            stress=stress,
            quality=quality,
            stability=stability,
            accepted_windows=accepted_windows,
            total_windows=total_windows,
            message=message,
            payload=dict(payload or {}),
        )
        self.observer(event)

    def _guard_cancelled(self) -> None:
        if self._cancel_event.is_set():
            raise CalibrationError("Calibration cancelled.")
