"""Subject-aware splitters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
from sklearn.model_selection import GroupKFold

from src.datasets.base import RawDatasetBundle, subset_bundle_by_subjects


@dataclass(slots=True)
class SubjectSplit:
    """Container for train/validation/test subject splits."""

    train: RawDatasetBundle
    val: RawDatasetBundle
    test: RawDatasetBundle
    train_subjects: list[str]
    val_subjects: list[str]
    test_subjects: list[str]


def train_val_test_subject_split(
    bundle: RawDatasetBundle,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> SubjectSplit:
    """Split a bundle by unique subject IDs."""
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("Split ratios must sum to 1.")
    subjects = sorted({record.subject_id for record in bundle.recordings})
    if len(subjects) < 3:
        raise ValueError("At least three subjects are required for train/val/test splitting.")

    rng = np.random.default_rng(seed)
    rng.shuffle(subjects)
    total = len(subjects)
    n_train = max(1, int(round(total * train_ratio)))
    n_val = max(1, int(round(total * val_ratio)))
    if n_train + n_val >= total:
        n_train = max(1, total - 2)
        n_val = 1
    n_test = total - n_train - n_val
    if n_test <= 0:
        n_test = 1
        if n_train > n_val:
            n_train -= 1
        else:
            n_val -= 1

    train_subjects = subjects[:n_train]
    val_subjects = subjects[n_train : n_train + n_val]
    test_subjects = subjects[n_train + n_val :]

    return SubjectSplit(
        train=subset_bundle_by_subjects(bundle, set(train_subjects)),
        val=subset_bundle_by_subjects(bundle, set(val_subjects)),
        test=subset_bundle_by_subjects(bundle, set(test_subjects)),
        train_subjects=train_subjects,
        val_subjects=val_subjects,
        test_subjects=test_subjects,
    )


def subject_kfold_splits(bundle: RawDatasetBundle, n_splits: int) -> Iterator[tuple[RawDatasetBundle, RawDatasetBundle]]:
    """Yield subject-aware k-fold splits."""
    subjects = np.array([record.subject_id for record in bundle.recordings])
    unique_subjects = sorted(set(subjects))
    if len(unique_subjects) < n_splits:
        raise ValueError("Number of unique subjects must be at least n_splits.")
    groups = subjects
    indices = np.arange(len(bundle.recordings))
    splitter = GroupKFold(n_splits=n_splits)
    for train_idx, val_idx in splitter.split(indices, groups=groups):
        train_subjects = {bundle.recordings[idx].subject_id for idx in train_idx}
        val_subjects = {bundle.recordings[idx].subject_id for idx in val_idx}
        yield subset_bundle_by_subjects(bundle, train_subjects), subset_bundle_by_subjects(bundle, val_subjects)
