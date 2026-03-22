"""Adaptive setup and runtime bundle tests."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.runtime.baseline import BaselinePrediction, BaselineInference
from src.runtime.calibration_controller import CalibrationController, CalibrationEvent, CalibrationPhaseConfig
from src.runtime.constants import CYTON_SAMPLE_RATE, RUNTIME_SETUP_CHANNELS
from src.runtime.contracts import DecisionScores, EEGChunk
from src.runtime.engine import StreamingEngine
from src.runtime.run_engine import validate_runtime_compatibility
from src.runtime.setup_interface import compute_stimulus_frame
from src.runtime.sources.base_source import EEGSource
from src.runtime.sources.synthetic_source import SyntheticSource
from src.runtime.user_profile import PhaseSummary, create_profile, load_user_profile, save_user_profile
from src.utils.io import load_pickle

REPO_ROOT = Path(__file__).resolve().parents[1]


class FakeAdaptiveSource(EEGSource):
    """Simple setup source whose condition is encoded directly in the signal."""

    source_name = "generator"

    def __init__(self, channel_names: list[str], chunk_samples: int = 125) -> None:
        self.channel_names = list(channel_names)
        self.chunk_samples = int(chunk_samples)
        self.timestamp = 0.0
        self.started = False
        self.concentration = 0.15
        self.stress = 0.10

    def set_condition(self, concentration: float, stress: float) -> None:
        self.concentration = float(concentration)
        self.stress = float(stress)

    def start(self) -> None:
        self.started = True
        self.timestamp = 0.0

    def stop(self) -> None:
        self.started = False

    def read_chunk(self) -> EEGChunk:
        if not self.started:
            raise RuntimeError("Source not started.")
        concentration_rows = np.full((4, self.chunk_samples), self.concentration, dtype=float)
        stress_rows = np.full((4, self.chunk_samples), self.stress, dtype=float)
        data = np.vstack([concentration_rows, stress_rows])
        chunk = EEGChunk(
            timestamp_start=self.timestamp,
            timestamp_end=self.timestamp + (self.chunk_samples / CYTON_SAMPLE_RATE),
            sample_rate=float(CYTON_SAMPLE_RATE),
            channel_names=list(self.channel_names),
            data=data,
            metadata={"source_name": self.source_name},
        )
        self.timestamp = chunk.timestamp_end
        return chunk


class FakeBaseline:
    """Decode concentration and stress scores directly from the fake source signal."""

    def predict_with_details(
        self,
        window: np.ndarray,
        sampling_rate: float = CYTON_SAMPLE_RATE,
        channel_names: list[str] | None = None,
    ) -> BaselinePrediction:
        concentration = float(np.clip(np.mean(window[:4]), 0.0, 1.0))
        stress = float(np.clip(np.mean(window[4:]), 0.0, 1.0))
        quality = 0.95
        return BaselinePrediction(
            scores=DecisionScores(
                concentration=concentration,
                stress=stress,
                confidence=quality,
                quality=quality,
            ),
            metadata={
                "concentration_score_100": concentration * 100.0,
                "concentration_probability": concentration,
                "stress_score_100": stress * 100.0,
                "stress_predicted_class": "high" if stress >= 0.66 else "natural",
                "quality_score_100": quality * 100.0,
                "quality_label": "clean",
                "artifact_probabilities": {"clean": quality},
                "artifact_flags": {"deterministic_window_passed": True},
            },
        )


def _short_protocol() -> list[CalibrationPhaseConfig]:
    return [
        CalibrationPhaseConfig("signal_check", "Signal Check", "usable_signal", "signal_check", 0.50, 2, 2, 4),
        CalibrationPhaseConfig("rest", "Rest", "rest", "rest", 0.50, 2, 2, 4, concentration_max=0.30, stress_max=0.20),
        CalibrationPhaseConfig("idle", "Idle", "idle", "idle", 0.50, 2, 2, 4, concentration_max=0.20, stress_max=0.20),
        CalibrationPhaseConfig(
            "focused",
            "Focus",
            "focused",
            "focused",
            0.50,
            2,
            2,
            14,
            adaptive=True,
            ramp_every_windows=2,
            max_modifier_level=4,
            concentration_min=0.83,
            stress_max=0.30,
            concentration_margin_over_stress=0.10,
        ),
        CalibrationPhaseConfig(
            "stressed",
            "Stress",
            "stressed",
            "stressed",
            0.50,
            2,
            2,
            14,
            adaptive=True,
            ramp_every_windows=2,
            max_modifier_level=4,
            stress_min=0.88,
        ),
        CalibrationPhaseConfig("recovery", "Recovery", "recovery", "recovery", 0.50, 2, 2, 4, concentration_max=0.30, stress_max=0.20),
    ]


def _phase_summary(name: str, concentration: float, stress: float) -> PhaseSummary:
    return PhaseSummary(
        phase_name=name,
        target_state=name,
        accepted=True,
        accepted_windows=6,
        total_windows=6,
        stability_score=1.0,
        quality_mean=0.95,
        concentration_quantiles={"p10": concentration - 0.02, "p50": concentration, "p90": concentration + 0.02},
        stress_quantiles={"p10": stress - 0.02, "p50": stress, "p90": stress + 0.02},
        feature_anchors={"f3_alpha": 0.5},
    )


def _make_runtime_signal(level: float, channels: list[str], samples: int = 768, sampling_rate: int = 128) -> np.ndarray:
    time_axis = np.arange(samples, dtype=float) / float(sampling_rate)
    rows = []
    for index, _ in enumerate(channels):
        alpha = np.sin(2.0 * np.pi * 10.0 * time_axis + index * 0.15)
        theta = np.sin(2.0 * np.pi * 6.0 * time_axis + index * 0.11)
        beta = np.sin(2.0 * np.pi * (18.0 + index * 0.3) * time_axis + index * 0.07)
        rows.append((1.0 - level) * alpha + level * beta + 0.25 * theta + 0.01 * (index + 1))
    return np.vstack(rows).astype(float)


def _write_runtime_fixture_roots(root: Path) -> dict[str, Path]:
    eegmat_root = root / "eegmat"
    stress_root = root / "stress"
    eegmat_root.mkdir(parents=True, exist_ok=True)
    stress_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "Subject": [f"Subject{index:02d}" for index in range(8)],
            "Age": [21 + index for index in range(8)],
            "Gender": ["F", "M"] * 4,
            "Recording year": [2011] * 8,
            "Number of subtractions": [10.0 + index for index in range(8)],
            "Count quality": [1] * 8,
        }
    ).to_csv(eegmat_root / "subject-info.csv", index=False)
    for index in range(8):
        np.savez(
            eegmat_root / f"Subject{index:02d}_1.npz",
            signal=_make_runtime_signal(0.18, RUNTIME_SETUP_CHANNELS),
            sampling_rate=128,
            channel_names=np.asarray(RUNTIME_SETUP_CHANNELS),
        )
        np.savez(
            eegmat_root / f"Subject{index:02d}_2.npz",
            signal=_make_runtime_signal(0.82, RUNTIME_SETUP_CHANNELS),
            sampling_rate=128,
            channel_names=np.asarray(RUNTIME_SETUP_CHANNELS),
        )

    rows: list[dict[str, object]] = []
    label_levels = {"natural": 0.10, "low": 0.35, "mid": 0.62, "high": 0.90}
    for index in range(8):
        subject_dir = stress_root / f"stress_subj_{index:02d}"
        subject_dir.mkdir(parents=True, exist_ok=True)
        for label, level in label_levels.items():
            file_path = subject_dir / f"{label}.npz"
            np.savez(
                file_path,
                signal=_make_runtime_signal(level, RUNTIME_SETUP_CHANNELS),
                sampling_rate=128,
                channel_names=np.asarray(RUNTIME_SETUP_CHANNELS),
            )
            rows.append(
                {
                    "file_path": str(file_path.relative_to(stress_root)),
                    "subject_id": f"stress_subj_{index:02d}",
                    "session_id": f"stress_subj_{index:02d}_{label}",
                    "label": label,
                    "sampling_rate": 128,
                    "channel_names": json.dumps(RUNTIME_SETUP_CHANNELS),
                    "target_score": {"natural": 0.0, "low": 0.33, "mid": 0.66, "high": 1.0}[label],
                }
            )
    pd.DataFrame(rows).to_csv(stress_root / "manifest.csv", index=False)
    return {"eegmat_root": eegmat_root, "stress_root": stress_root}


def test_calibration_controller_saves_profile_and_ramps_modifiers(tmp_path: Path) -> None:
    events: list[CalibrationEvent] = []
    controller = CalibrationController(
        source=FakeAdaptiveSource(list(RUNTIME_SETUP_CHANNELS)),
        source_type="generator",
        user_id="alice",
        artifacts_root=tmp_path / "artifacts",
        users_root=tmp_path / "users",
        expected_channels=list(RUNTIME_SETUP_CHANNELS),
        phase_plan=_short_protocol(),
        baseline=FakeBaseline(),
        observer=events.append,
    )

    result = controller.run()

    assert result.profile_files["profile_json"].exists()
    assert result.session_summary_path.exists()
    assert result.phase_results["focused"].accepted is True
    assert result.phase_results["stressed"].accepted is True
    assert result.user_profile.focus_high_anchor > result.user_profile.focus_low_anchor
    assert result.user_profile.stress_high_anchor > result.user_profile.stress_low_anchor

    loaded = load_user_profile(result.profile_files["profile_json"])
    assert loaded.user_id == "alice"
    assert loaded.channel_names == list(RUNTIME_SETUP_CHANNELS)
    assert any(event.kind == "session_completed" for event in events)


def test_streaming_engine_applies_user_profile_metadata(trained_artifacts: dict[str, Path], tmp_path: Path) -> None:
    profile = create_profile(
        user_id="bob",
        source_type="generator",
        channel_names=list(RUNTIME_SETUP_CHANNELS),
        phase_summaries={
            "rest": _phase_summary("rest", 0.18, 0.12),
            "idle": _phase_summary("idle", 0.22, 0.15),
            "focused": _phase_summary("focused", 0.82, 0.20),
            "stressed": _phase_summary("stressed", 0.34, 0.86),
        },
    )
    profile_files = save_user_profile(profile, tmp_path / "users")
    engine = StreamingEngine(
        artifacts_root=trained_artifacts["artifacts_root"],
        user_profile_path=profile_files["profile_json"],
        session_id="session_123",
    )
    source = SyntheticSource()
    source.start()
    outputs = []
    for _ in range(5):
        outputs.extend(engine.process_frame(source.read_frame()))
    source.stop()

    assert outputs
    metadata = outputs[-1].metadata
    assert metadata["profile_applied"] is True
    assert metadata["profile_id"] == profile.profile_id
    assert metadata["session_id"] == "session_123"
    assert 0.0 <= metadata["concentration_personalized"] <= 1.0
    assert 0.0 <= metadata["stress_personalized"] <= 1.0
    assert metadata["quadrant_state"] in {"idle", "focused", "stressed", "strained"}


def test_validate_runtime_setup_channels_rejects_old_artifacts(trained_artifacts: dict[str, Path]) -> None:
    baseline = BaselineInference(artifacts_root=trained_artifacts["artifacts_root"])
    with pytest.raises(ValueError, match="runtime stream"):
        validate_runtime_compatibility(baseline, runtime_channel_names=list(RUNTIME_SETUP_CHANNELS))


def test_compute_stimulus_frame_focus_and_stress_are_distinct() -> None:
    focus = compute_stimulus_frame("focused", elapsed_seconds=1.25, modifier_level=1)
    stress = compute_stimulus_frame("stressed", elapsed_seconds=1.25, modifier_level=1)

    assert focus.fill != stress.fill
    assert focus.jitter_x == pytest.approx(0.0)
    assert abs(stress.jitter_x) > 0.0
    assert focus.label in {"inhale", "exhale"}
    assert stress.label.isdigit()


def test_train_runtime_bundle_cli_produces_setup_channel_bundles(tmp_path: Path) -> None:
    roots = _write_runtime_fixture_roots(tmp_path / "data")
    artifacts_root = tmp_path / "runtime_v1"
    env = os.environ.copy()
    env.update(
        {
            "EEGMAT_ROOT": str(roots["eegmat_root"]),
            "STRESS_DATA_ROOT": str(roots["stress_root"]),
            "ARTIFACTS_ROOT": str(artifacts_root),
            "WINDOW_SECONDS": "2.0",
            "STRIDE_SECONDS": "1.0",
            "BANDPASS_LOW": "1.0",
            "BANDPASS_HIGH": "30.0",
            "NOTCH_FREQ": "",
            "AUTO_DOWNLOAD_EEGMAT": "false",
            "ENABLE_PYPREP": "false",
            "ENABLE_AUTOREJECT": "false",
            "RANDOM_SEED": "7",
        }
    )

    subprocess.run(
        [sys.executable, "-m", "src.training.train_runtime_bundle", "--artifacts-root", str(artifacts_root)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )

    concentration_preprocessor = load_pickle(artifacts_root / "concentration" / "preprocessor.pkl")
    stress_preprocessor = load_pickle(artifacts_root / "stress" / "preprocessor.pkl")
    summary = json.loads((artifacts_root / "runtime_bundle_summary.json").read_text(encoding="utf-8"))

    assert concentration_preprocessor.channel_names == list(RUNTIME_SETUP_CHANNELS)
    assert stress_preprocessor.channel_names == list(RUNTIME_SETUP_CHANNELS)
    assert summary["channel_names"] == list(RUNTIME_SETUP_CHANNELS)
