"""User calibration profile persistence and personalization helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np

from src.features.bandpower import compute_bandpower_summary
from src.utils.io import ensure_dir, load_json, load_pickle, save_json_data, save_pickle


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _quantile_triplet(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {"p10": 0.0, "p50": 0.0, "p90": 0.0}
    p10, p50, p90 = np.percentile(values, [10, 50, 90])
    return {"p10": float(p10), "p50": float(p50), "p90": float(p90)}


def _mean_feature_map(feature_rows: list[dict[str, float]]) -> dict[str, float]:
    if not feature_rows:
        return {}
    keys = sorted({key for row in feature_rows for key in row})
    return {
        key: float(np.mean([row.get(key, 0.0) for row in feature_rows], dtype=float))
        for key in keys
    }


def quadrant_state(concentration: float, stress: float, *, threshold: float = 0.6) -> str:
    focused = concentration >= threshold
    stressed = stress >= threshold
    if focused and stressed:
        return "strained"
    if focused:
        return "focused"
    if stressed:
        return "stressed"
    return "idle"


@dataclass(slots=True)
class PhaseSummary:
    """Compact calibration phase statistics saved into a user profile."""

    phase_name: str
    target_state: str
    accepted: bool
    accepted_windows: int
    total_windows: int
    stability_score: float
    quality_mean: float
    concentration_quantiles: dict[str, float]
    stress_quantiles: dict[str, float]
    feature_anchors: dict[str, float]
    modifiers: dict[str, float | int | str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class PersonalizationResult:
    """Profile-adjusted runtime output."""

    concentration_personalized: float
    stress_personalized: float
    quadrant_state: str
    profile_id: str
    session_id: str | None = None


@dataclass(slots=True)
class UserProfile:
    """Serializable user calibration profile for runtime personalization."""

    user_id: str
    profile_id: str
    created_at_utc: str
    source_type: str
    channel_names: list[str]
    focus_low_anchor: float
    focus_high_anchor: float
    stress_low_anchor: float
    stress_high_anchor: float
    personalization_threshold: float = 0.6
    smoothing_alpha: float = 0.35
    stress_semantics: str = "ui_stress_label_workload_proxy_v1"
    phase_summaries: dict[str, PhaseSummary] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def personalize(
        self,
        *,
        concentration_raw: float,
        stress_raw: float,
        session_id: str | None = None,
    ) -> PersonalizationResult:
        concentration = _normalize_between(concentration_raw, self.focus_low_anchor, self.focus_high_anchor)
        stress = _normalize_between(stress_raw, self.stress_low_anchor, self.stress_high_anchor)
        return PersonalizationResult(
            concentration_personalized=concentration,
            stress_personalized=stress,
            quadrant_state=quadrant_state(concentration, stress, threshold=self.personalization_threshold),
            profile_id=self.profile_id,
            session_id=session_id,
        )

    def to_json_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["phase_summaries"] = {
            key: asdict(value)
            for key, value in self.phase_summaries.items()
        }
        return payload


def _normalize_between(value: float, lower: float, upper: float) -> float:
    span = max(upper - lower, 1e-6)
    return float(np.clip((value - lower) / span, 0.0, 1.0))


def build_phase_feature_anchors(signal_windows: list[np.ndarray], sampling_rate: float, channel_names: list[str]) -> dict[str, float]:
    """Compute lightweight spectral anchors for one accepted phase."""
    if not signal_windows:
        return {}
    stacked = np.concatenate(signal_windows, axis=1)
    names = [name.upper() for name in channel_names]
    rows: list[dict[str, float]] = []
    for channel_index, channel_name in enumerate(names):
        absolute, _, _, ratios = compute_bandpower_summary(stacked[channel_index], sampling_rate)
        row = {
            f"{channel_name.lower()}_theta": absolute["theta"],
            f"{channel_name.lower()}_alpha": absolute["alpha"],
            f"{channel_name.lower()}_beta": absolute["beta"],
        }
        if channel_name in {"F3", "F4", "F7", "F8"}:
            row[f"{channel_name.lower()}_engagement_index"] = ratios["engagement_index"]
        rows.append(row)
    anchors = _mean_feature_map(rows)
    try:
        f3 = names.index("F3")
        f4 = names.index("F4")
    except ValueError:
        return anchors
    f3_alpha = compute_bandpower_summary(stacked[f3], sampling_rate)[0]["alpha"]
    f4_alpha = compute_bandpower_summary(stacked[f4], sampling_rate)[0]["alpha"]
    anchors["frontal_alpha_asymmetry"] = float(np.log1p(f4_alpha) - np.log1p(f3_alpha))
    return anchors


def profile_paths(root: Path, user_id: str) -> dict[str, Path]:
    user_root = root / user_id
    return {
        "user_root": user_root,
        "profile_json": user_root / "profile.json",
        "profile_pickle": user_root / "profile.pkl",
        "sessions_root": user_root / "sessions",
    }


def save_user_profile(profile: UserProfile, root: Path) -> dict[str, Path]:
    paths = profile_paths(root, profile.user_id)
    ensure_dir(paths["user_root"])
    save_json_data(profile.to_json_dict(), paths["profile_json"])
    save_pickle(profile, paths["profile_pickle"])
    return paths


def load_user_profile(path: Path) -> UserProfile:
    if path.suffix.lower() == ".json":
        payload = load_json(path)
        return UserProfile(
            **{
                **payload,
                "phase_summaries": {
                    key: PhaseSummary(**value)
                    for key, value in dict(payload.get("phase_summaries", {})).items()
                },
            }
        )
    profile = load_pickle(path)
    if not isinstance(profile, UserProfile):
        raise TypeError(f"Expected UserProfile at {path}, received {type(profile).__name__}.")
    return profile


def create_profile(
    *,
    user_id: str,
    source_type: str,
    channel_names: list[str],
    phase_summaries: dict[str, PhaseSummary],
    metadata: dict[str, Any] | None = None,
) -> UserProfile:
    focus_low_key = "concentration_low" if "concentration_low" in phase_summaries else "idle"
    focus_high_key = "concentration_high" if "concentration_high" in phase_summaries else "focused"
    stress_high_key = "stress_high" if "stress_high" in phase_summaries else "stressed"
    if "stress_low" in phase_summaries:
        stress_low = phase_summaries["stress_low"].stress_quantiles["p50"]
    else:
        stress_low = float(
            np.mean(
                [
                    phase_summaries["rest"].stress_quantiles["p50"],
                    phase_summaries["idle"].stress_quantiles["p50"],
                ],
                dtype=float,
            )
        )
    focus_low = phase_summaries[focus_low_key].concentration_quantiles["p50"]
    focus_high = phase_summaries[focus_high_key].concentration_quantiles["p50"]
    stress_high = phase_summaries[stress_high_key].stress_quantiles["p50"]
    return UserProfile(
        user_id=user_id,
        profile_id=f"profile_{uuid4().hex[:12]}",
        created_at_utc=_utc_timestamp(),
        source_type=source_type,
        channel_names=list(channel_names),
        focus_low_anchor=float(focus_low),
        focus_high_anchor=float(max(focus_high, focus_low + 1e-6)),
        stress_low_anchor=float(stress_low),
        stress_high_anchor=float(max(stress_high, stress_low + 1e-6)),
        phase_summaries=dict(phase_summaries),
        metadata=dict(metadata or {}),
    )


def save_phase_session(
    *,
    sessions_root: Path,
    session_id: str,
    phase_name: str,
    channel_names: list[str],
    sampling_rate: float,
    signals: list[np.ndarray],
    concentration_values: list[float],
    stress_values: list[float],
    quality_values: list[float],
    timestamps: list[float],
    modifiers: list[dict[str, float | int | str]],
) -> Path:
    session_root = ensure_dir(sessions_root / session_id)
    output_path = session_root / f"phase_{phase_name}.npz"
    np.savez(
        output_path,
        channel_names=np.asarray(channel_names),
        sampling_rate=float(sampling_rate),
        signals=np.asarray(signals, dtype=float),
        concentration_values=np.asarray(concentration_values, dtype=float),
        stress_values=np.asarray(stress_values, dtype=float),
        quality_values=np.asarray(quality_values, dtype=float),
        timestamps=np.asarray(timestamps, dtype=float),
        modifiers=np.asarray(modifiers, dtype=object),
    )
    return output_path


def summarize_phase(
    *,
    phase_name: str,
    target_state: str,
    accepted_windows: int,
    total_windows: int,
    concentration_values: list[float],
    stress_values: list[float],
    quality_values: list[float],
    feature_anchors: dict[str, float],
    modifiers: dict[str, float | int | str],
    accepted: bool,
    notes: list[str] | None = None,
) -> PhaseSummary:
    concentration_array = np.asarray(concentration_values, dtype=float)
    stress_array = np.asarray(stress_values, dtype=float)
    quality_array = np.asarray(quality_values, dtype=float)
    stability = float(accepted_windows / max(total_windows, 1))
    return PhaseSummary(
        phase_name=phase_name,
        target_state=target_state,
        accepted=bool(accepted),
        accepted_windows=int(accepted_windows),
        total_windows=int(total_windows),
        stability_score=stability,
        quality_mean=float(np.mean(quality_array)) if quality_array.size else 0.0,
        concentration_quantiles=_quantile_triplet(concentration_array),
        stress_quantiles=_quantile_triplet(stress_array),
        feature_anchors=dict(feature_anchors),
        modifiers=dict(modifiers),
        notes=list(notes or []),
    )
