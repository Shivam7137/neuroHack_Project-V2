"""Integration smoke tests for the CLI pipeline."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from src.inference.scorer import score_window


def test_training_cli_writes_expected_artifacts(trained_artifacts: dict[str, Path]) -> None:
    artifacts_root = trained_artifacts["artifacts_root"]
    assert (artifacts_root / "concentration" / "model.pkl").exists()
    assert (artifacts_root / "concentration" / "preprocessor.pkl").exists()
    assert (artifacts_root / "concentration" / "metrics_test.json").exists()
    assert (artifacts_root / "concentration" / "metrics_recording_test.json").exists()
    assert (artifacts_root / "concentration" / "metrics_pairwise_test.json").exists()
    assert (artifacts_root / "concentration" / "predictions_recording_test.csv").exists()
    assert (artifacts_root / "concentration" / "predictions_pairwise_test.csv").exists()
    assert (artifacts_root / "stress" / "model.pkl").exists()
    assert (artifacts_root / "stress" / "preprocessor.pkl").exists()
    assert (artifacts_root / "stress" / "metrics_test.json").exists()
    assert (artifacts_root / "stress" / "metrics_recording_test.json").exists()
    assert (artifacts_root / "stress" / "metrics_pairwise_test.json").exists()
    assert (artifacts_root / "stress" / "predictions_recording_test.csv").exists()
    assert (artifacts_root / "stress" / "predictions_pairwise_test.csv").exists()
    assert (artifacts_root / "artifact" / "model.pkl").exists()
    assert (artifacts_root / "artifact" / "preprocessor.pkl").exists()
    assert (artifacts_root / "artifact" / "metrics_test.json").exists()
    assert (artifacts_root / "artifact" / "metrics_recording_test.json").exists()
    assert (artifacts_root / "artifact" / "predictions_recording_test.csv").exists()
    assert (artifacts_root / "summary_all.json").exists()
    assert (artifacts_root / "stress" / "summary.json").exists()
    assert (artifacts_root / "artifact" / "summary.json").exists()

    concentration_summary = json.loads((artifacts_root / "concentration" / "summary.json").read_text(encoding="utf-8"))
    stress_summary = json.loads((artifacts_root / "stress" / "summary.json").read_text(encoding="utf-8"))
    artifact_summary = json.loads((artifacts_root / "artifact" / "summary.json").read_text(encoding="utf-8"))
    assert "test_recording" in concentration_summary
    assert "test_pairwise" in concentration_summary
    assert "test_recording" in stress_summary
    assert "test_pairwise" in stress_summary
    assert "test_recording" in artifact_summary


def test_inference_scores_one_window(trained_artifacts: dict[str, Path]) -> None:
    window_path = trained_artifacts["eegmat_root"] / "Subject00_2.npz"
    payload = np.load(window_path, allow_pickle=True)
    window = payload["signal"]
    result = score_window(
        window,
        sampling_rate=float(payload["sampling_rate"]),
        channel_names=[str(item) for item in payload["channel_names"].tolist()],
        artifacts_root=trained_artifacts["artifacts_root"],
    )
    assert 0.0 <= result["concentration_score"] <= 100.0
    assert 0.0 <= result["stress_score"] <= 100.0
    assert 0.0 <= result["quality_score"] <= 100.0
    assert isinstance(result["stress_class_probabilities"], dict)
    assert isinstance(result["artifact_probabilities"], dict)


def test_cli_scorer_command_outputs_json(trained_artifacts: dict[str, Path], tmp_path: Path) -> None:
    input_path = tmp_path / "window.npy"
    payload = np.load(trained_artifacts["eegmat_root"] / "Subject01_1.npz", allow_pickle=True)
    np.save(input_path, payload["signal"])
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.inference.scorer",
            "--input",
            str(input_path),
            "--sampling-rate",
            "128",
            "--channel-names",
            "C3,C4,P3,P4",
        ],
        cwd=Path(__file__).resolve().parents[1],
        env=trained_artifacts["env"],
        capture_output=True,
        text=True,
        check=True,
    )
    parsed = json.loads(completed.stdout)
    assert "concentration_score" in parsed
    assert "stress_score" in parsed
    assert "quality_score" in parsed
