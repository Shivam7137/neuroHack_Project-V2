"""Local-folder stress dataset loader."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import Settings, get_settings
from src.datasets.base import BaseDatasetLoader, RawDatasetBundle, RawRecording, WindowedDataset
from src.datasets.eegmat_loader import _read_edf_file
from src.preprocessing.windowing import create_windows

STRESS_LABEL_TO_CLASS = {"natural": 0, "low": 1, "mid": 2, "high": 3}
STRESS_LABEL_TO_SCORE = {"natural": 0.0, "low": 0.33, "mid": 0.66, "high": 1.0}
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


def _stew_rating_to_label(rating: int) -> str:
    """Map a 1-9 STEW self-rating into the ordinal stress bins."""
    if rating <= 2:
        return "natural"
    if rating <= 4:
        return "low"
    if rating <= 6:
        return "mid"
    return "high"


def _normalize_rating_score(rating: float) -> float:
    """Normalize a 1-9 subjective rating to [0, 1]."""
    return float(np.clip((rating - 1.0) / 8.0, 0.0, 1.0))


def _stew_condition_fallback(condition: str) -> str:
    """Fallback label mapping when STEW ratings are unavailable."""
    return {"lo": "low", "hi": "high"}[condition]


def _parse_channel_names(value: Any) -> list[str]:
    """Parse channel names from manifest values."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return []
        if value.startswith("["):
            return [str(item) for item in json.loads(value)]
        delimiter = "|" if "|" in value else ","
        return [part.strip() for part in value.split(delimiter) if part.strip()]
    return []


def _normalize_signal_shape(array: np.ndarray) -> np.ndarray:
    """Convert local file data into [channels, samples]."""
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D EEG array, received shape {array.shape}.")
    if array.shape[0] <= array.shape[1]:
        return array.astype(float)
    return array.T.astype(float)


class StressLocalLoader(BaseDatasetLoader):
    """Load a local OpenBCI/Cyton-like stress dataset."""

    dataset_name = "stress_local"

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def _load_signal(
        self,
        path: Path,
        sampling_rate: float | None,
        channel_names: list[str],
    ) -> tuple[np.ndarray, float, list[str]]:
        suffix = path.suffix.lower()
        if suffix == ".npy":
            signal = _normalize_signal_shape(np.load(path))
            return signal, float(sampling_rate or self.settings.stress_default_sampling_rate), channel_names or [
                f"ch_{idx:02d}" for idx in range(signal.shape[0])
            ]
        if suffix == ".npz":
            payload = np.load(path, allow_pickle=True)
            signal_key = "signal" if "signal" in payload else list(payload.keys())[0]
            signal = _normalize_signal_shape(payload[signal_key])
            file_sampling_rate = float(payload["sampling_rate"]) if "sampling_rate" in payload else float(
                sampling_rate or self.settings.stress_default_sampling_rate
            )
            file_channels = channel_names or (
                [str(item) for item in payload["channel_names"].tolist()]
                if "channel_names" in payload
                else [f"ch_{idx:02d}" for idx in range(signal.shape[0])]
            )
            return signal, file_sampling_rate, file_channels
        if suffix == ".csv":
            frame = pd.read_csv(path)
            numeric = frame.select_dtypes(include=["number"])
            signal = _normalize_signal_shape(numeric.to_numpy(dtype=float))
            inferred_channels = channel_names or [str(column) for column in numeric.columns] or [
                f"ch_{idx:02d}" for idx in range(signal.shape[0])
            ]
            return signal, float(sampling_rate or self.settings.stress_default_sampling_rate), inferred_channels
        if suffix == ".edf":
            signal, edf_sampling_rate, edf_channels = _read_edf_file(path)
            return signal, edf_sampling_rate, channel_names or edf_channels
        if suffix == ".txt":
            signal = _normalize_signal_shape(np.loadtxt(path, dtype=float))
            inferred_channels = channel_names or STEW_CHANNEL_NAMES[: signal.shape[0]]
            return signal, float(sampling_rate or self.settings.stress_default_sampling_rate), inferred_channels
        raise ValueError(f"Unsupported stress file format: {path.suffix}")

    def _infer_stew_manifest(self) -> pd.DataFrame | None:
        """Infer a manifest from the STEW dataset layout."""
        ratings_path = self.settings.stress_data_root / "ratings.txt"
        file_paths = sorted(self.settings.stress_data_root.glob("sub*_*.txt"))
        if not ratings_path.exists() or not file_paths:
            return None

        rating_map: dict[str, dict[str, int]] = {}
        for raw_line in ratings_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            subject_str, lo_str, hi_str = [part.strip() for part in line.split(",")]
            rating_map[f"sub{int(subject_str):02d}"] = {"lo": int(lo_str), "hi": int(hi_str)}

        rows: list[dict[str, Any]] = []
        for path in file_paths:
            stem_parts = path.stem.split("_")
            if len(stem_parts) != 2:
                continue
            subject_id, condition = stem_parts
            label = _stew_condition_fallback(condition)
            if subject_id in rating_map and condition in rating_map[subject_id]:
                label = _stew_rating_to_label(rating_map[subject_id][condition])
            rows.append(
                {
                    "file_path": str(path),
                    "subject_id": subject_id,
                    "session_id": path.stem,
                    "label": label,
                    "sampling_rate": 128,
                    "channel_names": json.dumps(STEW_CHANNEL_NAMES),
                    "source_variant": "stew",
                    "condition": condition,
                    "rating": rating_map.get(subject_id, {}).get(condition),
                    "rating_normalized": (
                        _normalize_rating_score(rating_map[subject_id][condition])
                        if subject_id in rating_map and condition in rating_map[subject_id]
                        else None
                    ),
                }
            )
        return pd.DataFrame(rows) if rows else None

    def _infer_manifest(self) -> pd.DataFrame:
        stew_manifest = self._infer_stew_manifest()
        if stew_manifest is not None:
            return stew_manifest

        rows: list[dict[str, Any]] = []
        for label in STRESS_LABEL_TO_CLASS:
            label_root = self.settings.stress_data_root / label
            if not label_root.exists():
                continue
            for path in sorted(label_root.rglob("*")):
                if path.is_file() and path.suffix.lower() in self.settings.supported_stress_suffixes:
                    subject_id = path.parent.name if path.parent != label_root else path.stem.split("_")[0]
                    rows.append(
                        {
                            "file_path": str(path),
                            "subject_id": subject_id,
                            "session_id": path.stem,
                            "label": label,
                        }
                    )
        if not rows:
            raise FileNotFoundError(
                f"No stress data found under {self.settings.stress_data_root}. "
                "Provide a manifest.csv or label folders named natural/low/mid/high."
            )
        return pd.DataFrame(rows)

    def load_raw(self) -> RawDatasetBundle:
        """Load local stress recordings from a manifest or label folders."""
        manifest_path = self.settings.stress_data_root / "manifest.csv"
        manifest = pd.read_csv(manifest_path) if manifest_path.exists() else self._infer_manifest()
        required = {"file_path", "subject_id", "label"}
        missing = required - set(manifest.columns)
        if missing:
            raise ValueError(f"Stress manifest is missing required columns: {sorted(missing)}")

        recordings: list[RawRecording] = []
        for row in manifest.to_dict(orient="records"):
            relative_path = Path(str(row["file_path"]))
            file_path = relative_path if relative_path.is_absolute() else (self.settings.stress_data_root / relative_path)
            if not file_path.exists():
                raise FileNotFoundError(f"Stress recording file not found: {file_path}")
            channel_names = _parse_channel_names(row.get("channel_names"))
            signal, sampling_rate, resolved_channels = self._load_signal(
                file_path,
                sampling_rate=float(row["sampling_rate"]) if row.get("sampling_rate") not in {None, ""} else None,
                channel_names=channel_names,
            )
            recordings.append(
                RawRecording(
                    signal=signal,
                    sampling_rate=sampling_rate,
                    channel_names=resolved_channels,
                    subject_id=str(row["subject_id"]),
                    session_id=str(row["session_id"]) if row.get("session_id") not in {None, ""} else file_path.stem,
                    source_dataset=self.dataset_name,
                    raw_label=str(row["label"]).lower().strip(),
                    mapped_label=None,
                    extra_metadata={
                        "file_path": str(file_path),
                        **{
                            key: row[key]
                            for key in ("source_variant", "condition", "rating", "rating_normalized")
                            if key in row and row[key] not in {None, ""}
                        },
                    },
                )
            )
        return self.map_labels(RawDatasetBundle.from_recordings(recordings))

    def map_labels(self, bundle: RawDatasetBundle) -> RawDatasetBundle:
        """Map stress labels into ordinal class IDs."""
        remapped: list[RawRecording] = []
        for record in bundle.recordings:
            raw_label = record.raw_label.lower().strip()
            if raw_label not in STRESS_LABEL_TO_CLASS:
                raise ValueError(f"Unsupported stress label '{record.raw_label}'. Expected one of {sorted(STRESS_LABEL_TO_CLASS)}.")
            remapped.append(
                RawRecording(
                    signal=record.signal,
                    sampling_rate=record.sampling_rate,
                    channel_names=record.channel_names,
                    subject_id=record.subject_id,
                    session_id=record.session_id,
                    source_dataset=record.source_dataset,
                    raw_label=raw_label,
                    mapped_label=STRESS_LABEL_TO_CLASS[raw_label],
                    extra_metadata={
                        **record.extra_metadata,
                        "mapped_score": STRESS_LABEL_TO_SCORE[raw_label],
                        "target_score": record.extra_metadata.get("rating_normalized", STRESS_LABEL_TO_SCORE[raw_label]),
                    },
                )
            )
        return RawDatasetBundle.from_recordings(remapped)

    def make_windows(self, bundle: RawDatasetBundle, preprocessor: Any) -> WindowedDataset:
        """Window stress recordings with a fitted preprocessor."""
        windows: list[np.ndarray] = []
        labels: list[int] = []
        metadata_rows: list[dict[str, Any]] = []
        for record in bundle.recordings:
            processed = preprocessor.transform_raw(record.signal, record.sampling_rate, record.channel_names)
            effective_sampling_rate = float(preprocessor.target_sampling_rate or record.sampling_rate)
            recording_windows, bounds = create_windows(
                processed,
                sampling_rate=effective_sampling_rate,
                window_seconds=self.settings.window_seconds,
                stride_seconds=self.settings.stride_seconds,
            )
            for index, (window, (start, stop)) in enumerate(zip(recording_windows, bounds, strict=True)):
                windows.append(window)
                labels.append(int(record.mapped_label))
                metadata_rows.append(
                    {
                        "subject_id": record.subject_id,
                        "session_id": record.session_id,
                        "source_dataset": record.source_dataset,
                        "raw_label": record.raw_label,
                        "mapped_label": record.mapped_label,
                        "mapped_score": STRESS_LABEL_TO_SCORE[record.raw_label],
                        "sampling_rate": effective_sampling_rate,
                        "channel_names": "|".join(preprocessor.channel_names),
                        "window_index": index,
                        "start_sample": start,
                        "end_sample": stop,
                        **record.extra_metadata,
                    }
                )
        if not windows:
            return WindowedDataset(
                windows=np.empty((0, 0, 0), dtype=float),
                labels=np.empty((0,), dtype=int),
                metadata=pd.DataFrame(metadata_rows),
            )
        return WindowedDataset(
            windows=np.stack(windows, axis=0),
            labels=np.asarray(labels, dtype=int),
            metadata=pd.DataFrame(metadata_rows),
        )
