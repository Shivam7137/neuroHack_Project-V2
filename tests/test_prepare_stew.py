"""Integration tests for the dedicated STEW preparation CLI."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def test_prepare_stew_cli_writes_expected_artifacts(synthetic_stew_root: Path, tmp_path: Path) -> None:
    artifacts_root = tmp_path / "artifacts"
    env = os.environ.copy()
    env.update(
        {
            "STRESS_DATA_ROOT": str(synthetic_stew_root),
            "ARTIFACTS_ROOT": str(artifacts_root),
            "WINDOW_SECONDS": "2.0",
            "STRIDE_SECONDS": "0.5",
            "BANDPASS_LOW": "1.0",
            "BANDPASS_HIGH": "40.0",
            "NOTCH_FREQ": "",
            "RANDOM_SEED": "11",
        }
    )

    completed = subprocess.run(
        [sys.executable, "-m", "src.training.prepare_stew"],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )

    stew_root = artifacts_root / "stew"
    expected_files = [
        "metadata_recordings.csv",
        "metadata_windows.csv",
        "subject_split.csv",
        "X_train.npy",
        "X_val.npy",
        "X_test.npy",
        "y_train_binary.npy",
        "y_val_binary.npy",
        "y_test_binary.npy",
        "y_train_ordinal.npy",
        "y_val_ordinal.npy",
        "y_test_ordinal.npy",
        "feature_names.json",
        "preprocessing.pkl",
        "prep_summary.json",
    ]
    for file_name in expected_files:
        assert (stew_root / file_name).exists(), file_name

    metadata_windows = pd.read_csv(stew_root / "metadata_windows.csv")
    subject_split = pd.read_csv(stew_root / "subject_split.csv")
    X_train = np.load(stew_root / "X_train.npy")
    X_val = np.load(stew_root / "X_val.npy")
    X_test = np.load(stew_root / "X_test.npy")
    feature_names = json.loads((stew_root / "feature_names.json").read_text(encoding="utf-8"))
    summary = json.loads((stew_root / "prep_summary.json").read_text(encoding="utf-8"))

    assert X_train.shape[0] == int((metadata_windows["split"] == "train").sum())
    assert X_val.shape[0] == int((metadata_windows["split"] == "val").sum())
    assert X_test.shape[0] == int((metadata_windows["split"] == "test").sum())
    assert X_train.shape[1] == len(feature_names)
    assert summary["number_of_subjects"] == int(subject_split["subject_id"].nunique())
    assert "stew preparation complete" in completed.stdout.lower()


def test_prepare_stew_without_ratings_uses_binary_fallback(tmp_path: Path) -> None:
    stew_root = tmp_path / "stew"
    stew_root.mkdir(parents=True, exist_ok=True)
    for subject_idx in range(3):
        subject_id = f"sub{subject_idx + 1:02d}"
        np.savetxt(stew_root / f"{subject_id}_lo.txt", np.ones((256, 14)), fmt="%.4f")
        np.savetxt(stew_root / f"{subject_id}_hi.txt", np.ones((256, 14)) * 2.0, fmt="%.4f")

    artifacts_root = tmp_path / "artifacts"
    env = os.environ.copy()
    env.update(
        {
            "STRESS_DATA_ROOT": str(stew_root),
            "ARTIFACTS_ROOT": str(artifacts_root),
            "RANDOM_SEED": "3",
        }
    )
    subprocess.run(
        [sys.executable, "-m", "src.training.prepare_stew"],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )

    metadata_windows = pd.read_csv(artifacts_root / "stew" / "metadata_windows.csv")
    assert set(metadata_windows["ordinal_label"]) == {"low", "high"}
    assert set(np.unique(np.load(artifacts_root / "stew" / "y_train_ordinal.npy"))).issubset({0.0, 1.0})
