"""Canonical window dataset builder for synthetic EEG training."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.signal import resample_poly

from src.datasets.base import RawDatasetBundle
from src.preprocessing.normalization import canonicalize_channel_names
from src.preprocessing.windowing import create_windows


def _expand_channels(
    signal: np.ndarray,
    channel_names: list[str],
    target_channel_names: list[str],
) -> np.ndarray:
    current = canonicalize_channel_names(channel_names)
    target = canonicalize_channel_names(target_channel_names)
    expanded = np.zeros((len(target), signal.shape[1]), dtype=float)
    index_map = {name: idx for idx, name in enumerate(current)}
    for target_index, name in enumerate(target):
        if name in index_map:
            expanded[target_index] = signal[index_map[name]]
    return expanded


@dataclass(slots=True)
class CanonicalWindowDataset:
    """Canonical fixed windows for generator training."""

    windows: np.ndarray
    metadata: pd.DataFrame
    channel_names: list[str]
    sample_rate: float


def build_canonical_window_dataset(
    bundle: RawDatasetBundle,
    target_channel_names: list[str],
    target_sampling_rate: float = 250.0,
    window_seconds: float = 2.0,
    stride_seconds: float = 0.5,
) -> CanonicalWindowDataset:
    rows: list[np.ndarray] = []
    metadata_rows: list[dict[str, object]] = []
    for record in bundle.recordings:
        expanded = _expand_channels(record.signal, record.channel_names, target_channel_names)
        if not np.isclose(record.sampling_rate, target_sampling_rate):
            expanded = resample_poly(
                expanded,
                up=int(round(target_sampling_rate)),
                down=int(round(record.sampling_rate)),
                axis=-1,
            )
        windows, bounds = create_windows(
            expanded,
            sampling_rate=target_sampling_rate,
            window_seconds=window_seconds,
            stride_seconds=stride_seconds,
        )
        for index, (window, (start, stop)) in enumerate(zip(windows, bounds, strict=True)):
            rows.append(window.astype(float))
            metadata_rows.append(
                {
                    "subject_id": record.subject_id,
                    "session_id": record.session_id,
                    "source_dataset": record.source_dataset,
                    "raw_label": record.raw_label,
                    "mapped_label": record.mapped_label,
                    "window_index": index,
                    "start_sample": start,
                    "stop_sample": stop,
                    **record.extra_metadata,
                }
            )
    return CanonicalWindowDataset(
        windows=np.stack(rows, axis=0) if rows else np.empty((0, len(target_channel_names), 0), dtype=float),
        metadata=pd.DataFrame(metadata_rows),
        channel_names=list(target_channel_names),
        sample_rate=float(target_sampling_rate),
    )
