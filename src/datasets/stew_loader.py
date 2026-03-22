"""Dedicated STEW workload dataset loader."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.datasets.base import BaseDatasetLoader, RawDatasetBundle, RawRecording, WindowedDataset
from src.preprocessing.windowing import create_windows
from src.utils.logging_utils import get_logger

LOGGER = get_logger("datasets.stew_loader")

STEW_CHANNEL_NAMES = [
    "AF3",
    "F7",
    "F3",
    "FC5",
    "T7",
    "P7",
    "O1",
    "O2",
    "P8",
    "T8",
    "FC6",
    "F4",
    "F8",
    "AF4",
]
STEW_FILENAME_PATTERN = re.compile(r"^(sub\d+)_(lo|hi)$", re.IGNORECASE)


def _binary_label_from_condition(condition_raw: str) -> int:
    return 0 if condition_raw == "lo" else 1


def _ordinal_from_rating(rating: int) -> tuple[str, float]:
    if 1 <= rating <= 3:
        return "low", 0.0
    if 4 <= rating <= 6:
        return "medium", 0.5
    if 7 <= rating <= 9:
        return "high", 1.0
    raise ValueError(f"STEW workload ratings must be in [1, 9], received {rating}.")


def _fallback_ordinal(condition_raw: str) -> tuple[str, float]:
    return ("low", 0.0) if condition_raw == "lo" else ("high", 1.0)


@dataclass(slots=True)
class DiscoveredSTEWFile:
    """One discovered STEW recording path."""

    path: Path
    subject_id: str
    condition_raw: str


@dataclass(slots=True)
class STEWLoader(BaseDatasetLoader):
    """Load the local STEW dataset as a workload/concentration corpus."""

    data_root: str | Path
    dataset_name: str = "STEW"
    sampling_rate: float = 128.0
    discovered_files_cache: list[DiscoveredSTEWFile] = field(default_factory=list)
    malformed_files: list[dict[str, str]] = field(default_factory=list)
    skipped_files: list[dict[str, str]] = field(default_factory=list)
    ratings_lookup: dict[tuple[str, str], int] = field(default_factory=dict)
    _bundle_cache: RawDatasetBundle | None = None

    def __post_init__(self) -> None:
        self.data_root = Path(self.data_root).expanduser().resolve()

    def discover_files(self) -> list[DiscoveredSTEWFile]:
        """Recursively find valid STEW text recordings."""
        discovered: list[DiscoveredSTEWFile] = []
        skipped: list[dict[str, str]] = []
        for path in sorted(self.data_root.rglob("*.txt")):
            if path.name.lower() == "ratings.txt":
                continue
            match = STEW_FILENAME_PATTERN.match(path.stem)
            if match is None:
                skipped.append({"file_path": str(path), "reason": "filename_pattern_mismatch"})
                continue
            subject_id, condition_raw = match.groups()
            discovered.append(
                DiscoveredSTEWFile(
                    path=path,
                    subject_id=subject_id.lower(),
                    condition_raw=condition_raw.lower(),
                )
            )
        if not discovered:
            raise FileNotFoundError(
                f"No STEW files matching sub*_lo.txt or sub*_hi.txt were found under {self.data_root}."
            )
        self.discovered_files_cache = discovered
        self.skipped_files = skipped
        LOGGER.info(
            "Discovered %s STEW files across %s subjects under %s",
            len(discovered),
            len({item.subject_id for item in discovered}),
            self.data_root,
        )
        if skipped:
            LOGGER.warning("Skipped %s non-STEW text files during discovery", len(skipped))
        return list(discovered)

    def _load_ratings(self) -> dict[tuple[str, str], int]:
        """Load optional STEW workload ratings."""
        ratings_path = self.data_root / "ratings.txt"
        if not ratings_path.exists():
            LOGGER.info("No ratings.txt found under %s; using lo/hi fallback ordinal targets.", self.data_root)
            self.ratings_lookup = {}
            return {}

        lookup: dict[tuple[str, str], int] = {}
        for line_number, raw_line in enumerate(ratings_path.read_text(encoding="utf-8").splitlines(), start=1):
            line = raw_line.strip()
            if not line:
                continue
            parts = [part.strip() for part in line.split(",")]
            if len(parts) != 3:
                raise ValueError(f"Malformed ratings.txt line {line_number}: expected 'subject, low, high', received {line!r}.")
            subject_id = f"sub{int(parts[0]):02d}"
            lookup[(subject_id, "lo")] = int(parts[1])
            lookup[(subject_id, "hi")] = int(parts[2])
        self.ratings_lookup = lookup
        LOGGER.info("Loaded workload ratings for %s STEW condition files", len(lookup))
        return lookup

    def _load_matrix(self, path: Path) -> np.ndarray:
        """Read one STEW matrix and convert it to [channels, samples]."""
        matrix = np.loadtxt(path, dtype=float)
        if matrix.ndim != 2:
            raise ValueError(f"Expected a 2D matrix, received shape {matrix.shape}.")
        if matrix.shape[1] == len(STEW_CHANNEL_NAMES):
            return matrix.T.astype(float)
        if matrix.shape[0] == len(STEW_CHANNEL_NAMES):
            LOGGER.warning("STEW file %s appears transposed already; using it as [channels, samples].", path)
            return matrix.astype(float)
        raise ValueError(f"Expected 14 channels in STEW file {path}, received shape {matrix.shape}.")

    def load_subject_recordings(self) -> list[RawRecording]:
        """Load usable STEW recordings into RawRecording objects."""
        discovered = self.discover_files() if not self.discovered_files_cache else list(self.discovered_files_cache)
        ratings = self._load_ratings()
        malformed: list[dict[str, str]] = []
        recordings: list[RawRecording] = []

        for item in discovered:
            try:
                signal = self._load_matrix(item.path)
            except Exception as exc:  # noqa: BLE001
                malformed.append({"file_path": str(item.path), "reason": str(exc)})
                LOGGER.warning("Skipping malformed STEW file %s: %s", item.path, exc)
                continue

            workload_rating = ratings.get((item.subject_id, item.condition_raw))
            ordinal_label, ordinal_target = (
                _ordinal_from_rating(workload_rating)
                if workload_rating is not None
                else _fallback_ordinal(item.condition_raw)
            )
            recordings.append(
                RawRecording(
                    signal=signal,
                    sampling_rate=self.sampling_rate,
                    channel_names=list(STEW_CHANNEL_NAMES),
                    subject_id=item.subject_id,
                    session_id="0",
                    source_dataset=self.dataset_name,
                    raw_label=item.condition_raw,
                    mapped_label=_binary_label_from_condition(item.condition_raw),
                    extra_metadata={
                        "file_path": str(item.path),
                        "condition_raw": item.condition_raw,
                        "binary_label": _binary_label_from_condition(item.condition_raw),
                        "ordinal_label": ordinal_label,
                        "ordinal_target": ordinal_target,
                        "workload_rating": workload_rating,
                        "n_samples": int(signal.shape[1]),
                        "duration_seconds": float(signal.shape[1] / self.sampling_rate),
                    },
                )
            )

        self.malformed_files = malformed
        if not recordings:
            raise FileNotFoundError(f"No valid STEW recordings could be loaded from {self.data_root}.")
        LOGGER.info("Loaded %s valid STEW recordings; malformed=%s", len(recordings), len(malformed))
        return recordings

    def build_metadata(self) -> pd.DataFrame:
        """Return recording-level metadata."""
        bundle = self.load_raw()
        return bundle.metadata.copy()

    def load_raw(self) -> RawDatasetBundle:
        """Load raw STEW recordings and aligned metadata."""
        if self._bundle_cache is None:
            self._bundle_cache = self.map_labels(RawDatasetBundle.from_recordings(self.load_subject_recordings()))
        return self._bundle_cache

    def map_labels(self, bundle: RawDatasetBundle) -> RawDatasetBundle:
        """Preserve binary labels as the main mapped label."""
        remapped: list[RawRecording] = []
        for record in bundle.recordings:
            remapped.append(
                RawRecording(
                    signal=record.signal,
                    sampling_rate=record.sampling_rate,
                    channel_names=record.channel_names,
                    subject_id=record.subject_id,
                    session_id=record.session_id,
                    source_dataset=record.source_dataset,
                    raw_label=record.raw_label,
                    mapped_label=int(record.extra_metadata["binary_label"]),
                    extra_metadata=dict(record.extra_metadata),
                )
            )
        return RawDatasetBundle.from_recordings(remapped)

    def make_windows(self, bundle: RawDatasetBundle, preprocessor: Any) -> WindowedDataset:
        """Window STEW recordings after preprocessing."""
        windows: list[np.ndarray] = []
        labels: list[int] = []
        metadata_rows: list[dict[str, Any]] = []
        for record in bundle.recordings:
            processed = preprocessor.transform_raw(record.signal, record.sampling_rate, record.channel_names)
            recording_windows, bounds = create_windows(
                processed,
                sampling_rate=preprocessor.sampling_rate,
                window_seconds=preprocessor.window_seconds,
                stride_seconds=preprocessor.stride_seconds,
            )
            for window_index, (window, (start_sample, end_sample)) in enumerate(zip(recording_windows, bounds, strict=True)):
                windows.append(window)
                labels.append(int(record.extra_metadata["binary_label"]))
                metadata_rows.append(
                    {
                        "subject_id": record.subject_id,
                        "session_id": record.session_id,
                        "source_dataset": record.source_dataset,
                        "source_file": record.extra_metadata["file_path"],
                        "condition_raw": record.extra_metadata["condition_raw"],
                        "binary_label": int(record.extra_metadata["binary_label"]),
                        "ordinal_label": record.extra_metadata["ordinal_label"],
                        "ordinal_target": float(record.extra_metadata["ordinal_target"]),
                        "workload_rating": record.extra_metadata.get("workload_rating"),
                        "sampling_rate": preprocessor.sampling_rate,
                        "channel_names": "|".join(preprocessor.channel_names),
                        "window_index": window_index,
                        "start_sample": start_sample,
                        "end_sample": end_sample,
                    }
                )
        if not windows:
            return WindowedDataset(
                windows=np.empty((0, len(preprocessor.channel_names), 0), dtype=float),
                labels=np.empty((0,), dtype=int),
                metadata=pd.DataFrame(metadata_rows),
            )
        return WindowedDataset(
            windows=np.stack(windows, axis=0),
            labels=np.asarray(labels, dtype=int),
            metadata=pd.DataFrame(metadata_rows),
        )
