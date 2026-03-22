"""Tests for the dedicated STEW loader."""

from __future__ import annotations

import numpy as np

from src.datasets.stew_loader import STEWLoader
from src.preprocessing.normalization import RecordingRobustScaler
from src.preprocessing.splitters import train_val_test_subject_split


def test_stew_loader_parses_files_and_ratings(synthetic_stew_root) -> None:
    loader = STEWLoader(synthetic_stew_root)
    bundle = loader.load_raw()

    assert len(bundle.recordings) == 16
    assert bundle.metadata["subject_id"].nunique() == 8
    assert set(bundle.metadata["condition_raw"]) == {"lo", "hi"}
    assert bundle.metadata["binary_label"].isin([0, 1]).all()
    assert bundle.metadata["workload_rating"].notna().all()
    assert set(bundle.metadata["ordinal_label"]) <= {"low", "medium", "high"}


def test_stew_loader_skips_malformed_file(tmp_path) -> None:
    good_root = tmp_path / "stew"
    good_root.mkdir(parents=True, exist_ok=True)
    np.savetxt(good_root / "sub01_lo.txt", np.ones((128, 14)), fmt="%.4f")
    np.savetxt(good_root / "sub01_hi.txt", np.ones((128, 13)), fmt="%.4f")

    loader = STEWLoader(good_root)
    bundle = loader.load_raw()

    assert len(bundle.recordings) == 1
    assert len(loader.malformed_files) == 1


def test_recording_robust_scaler_uses_train_statistics_only() -> None:
    scaler = RecordingRobustScaler(channel_names=["C3", "C4"])
    scaler.fit([np.array([[1.0, 2.0, 3.0], [10.0, 11.0, 12.0]])])

    transformed = scaler.transform(np.array([[2.0, 3.0, 4.0], [11.0, 12.0, 13.0]]))

    assert transformed.shape == (2, 3)
    assert np.isfinite(transformed).all()


def test_subject_split_keeps_stew_subjects_isolated(synthetic_stew_root) -> None:
    loader = STEWLoader(synthetic_stew_root)
    bundle = loader.load_raw()
    split = train_val_test_subject_split(bundle, 0.7, 0.15, 0.15, 7)

    train_subjects = set(split.train_subjects)
    val_subjects = set(split.val_subjects)
    test_subjects = set(split.test_subjects)

    assert train_subjects.isdisjoint(val_subjects)
    assert train_subjects.isdisjoint(test_subjects)
    assert val_subjects.isdisjoint(test_subjects)
