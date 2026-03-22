"""Tests for subject-aware splitting."""

from __future__ import annotations

import numpy as np

from src.datasets.base import RawDatasetBundle, RawRecording
from src.preprocessing.splitters import train_val_test_subject_split


def test_subject_split_has_no_overlap() -> None:
    recordings = []
    for subject_idx in range(10):
        recordings.append(
            RawRecording(
                signal=np.zeros((2, 16)),
                sampling_rate=128.0,
                channel_names=["C3", "C4"],
                subject_id=f"s{subject_idx}",
                session_id="1",
                source_dataset="unit",
                raw_label="baseline",
                mapped_label=0,
            )
        )
    bundle = RawDatasetBundle.from_recordings(recordings)
    split = train_val_test_subject_split(bundle, 0.7, 0.15, 0.15, seed=1)
    assert not (set(split.train_subjects) & set(split.val_subjects))
    assert not (set(split.train_subjects) & set(split.test_subjects))
    assert not (set(split.val_subjects) & set(split.test_subjects))
