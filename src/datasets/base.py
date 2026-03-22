"""Shared dataset abstractions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass(slots=True)
class RawRecording:
    """One continuous or pre-windowed EEG recording."""

    signal: np.ndarray
    sampling_rate: float
    channel_names: list[str]
    subject_id: str
    session_id: str | None
    source_dataset: str
    raw_label: str
    mapped_label: float | int | None
    extra_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RawDatasetBundle:
    """Collection of raw recordings plus tabular metadata."""

    recordings: list[RawRecording]
    metadata: pd.DataFrame

    @classmethod
    def from_recordings(cls, recordings: list[RawRecording]) -> "RawDatasetBundle":
        """Build a bundle and metadata frame from recordings."""
        rows: list[dict[str, Any]] = []
        for record in recordings:
            rows.append(
                {
                    "subject_id": record.subject_id,
                    "session_id": record.session_id,
                    "source_dataset": record.source_dataset,
                    "raw_label": record.raw_label,
                    "mapped_label": record.mapped_label,
                    "sampling_rate": record.sampling_rate,
                    "channel_names": "|".join(record.channel_names),
                    **record.extra_metadata,
                }
            )
        metadata = pd.DataFrame(rows)
        return cls(recordings=recordings, metadata=metadata)


@dataclass(slots=True)
class WindowedDataset:
    """Windowed EEG arrays, labels, and aligned metadata."""

    windows: np.ndarray
    labels: np.ndarray
    metadata: pd.DataFrame


class BaseDatasetLoader(ABC):
    """Base interface for raw EEG dataset loaders."""

    dataset_name: str

    @abstractmethod
    def load_raw(self) -> RawDatasetBundle:
        """Load recordings from disk or remote source."""

    @abstractmethod
    def map_labels(self, bundle: RawDatasetBundle) -> RawDatasetBundle:
        """Map raw labels into the task target space."""

    @abstractmethod
    def make_windows(self, bundle: RawDatasetBundle, preprocessor: Any) -> WindowedDataset:
        """Convert recordings into fixed-length windows."""


def subset_bundle_by_subjects(bundle: RawDatasetBundle, subject_ids: set[str]) -> RawDatasetBundle:
    """Return a bundle filtered to a subject subset."""
    selected = [record for record in bundle.recordings if record.subject_id in subject_ids]
    return RawDatasetBundle.from_recordings(selected)
