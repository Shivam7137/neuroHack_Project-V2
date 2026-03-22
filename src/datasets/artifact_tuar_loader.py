"""Local TUAR-style artifact dataset loader."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import Settings, get_settings
from src.datasets.artifact_common import artifact_label_id
from src.datasets.base import BaseDatasetLoader, RawDatasetBundle, RawRecording, WindowedDataset
from src.datasets.eegmat_loader import _read_edf_file
from src.datasets.stress_local_loader import _normalize_signal_shape, _parse_channel_names
from src.preprocessing.windowing import create_windows


class TUARLoader(BaseDatasetLoader):
    """Load artifact recordings from a TUAR-style manifest or folder structure."""

    dataset_name = "tuar"

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
            return signal, float(sampling_rate or 256.0), channel_names or [f"ch_{idx:02d}" for idx in range(signal.shape[0])]
        if suffix == ".npz":
            payload = np.load(path, allow_pickle=True)
            signal_key = "signal" if "signal" in payload else list(payload.keys())[0]
            signal = _normalize_signal_shape(np.asarray(payload[signal_key], dtype=float))
            file_sampling_rate = float(payload["sampling_rate"]) if "sampling_rate" in payload else float(sampling_rate or 256.0)
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
            inferred_channels = channel_names or [str(column) for column in numeric.columns] or [f"ch_{idx:02d}" for idx in range(signal.shape[0])]
            return signal, float(sampling_rate or 256.0), inferred_channels
        if suffix == ".edf":
            signal, edf_sampling_rate, edf_channels = _read_edf_file(path)
            return signal, edf_sampling_rate, channel_names or edf_channels
        raise ValueError(f"Unsupported TUAR file format: {path.suffix}")

    def _infer_manifest(self) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for label_dir in sorted(self.settings.tuar_root.iterdir()):
            if not label_dir.is_dir():
                continue
            label = label_dir.name
            for path in sorted(label_dir.rglob("*")):
                if not path.is_file() or path.suffix.lower() not in {".npy", ".npz", ".csv", ".edf"}:
                    continue
                subject_id = path.parent.name if path.parent != label_dir else path.stem.split("_")[0]
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
                f"No TUAR artifact data found under {self.settings.tuar_root}. "
                "Provide a manifest.csv or label folders named clean/eyem/chew/shiv/elpp/musc."
            )
        return pd.DataFrame(rows)

    def load_raw(self) -> RawDatasetBundle:
        """Load artifact recordings from disk."""
        manifest_path = self.settings.tuar_root / "manifest.csv"
        manifest = pd.read_csv(manifest_path) if manifest_path.exists() else self._infer_manifest()
        required = {"file_path", "subject_id", "label"}
        missing = required - set(manifest.columns)
        if missing:
            raise ValueError(f"TUAR manifest is missing required columns: {sorted(missing)}")

        recordings: list[RawRecording] = []
        for row in manifest.to_dict(orient="records"):
            relative_path = Path(str(row["file_path"]))
            file_path = relative_path if relative_path.is_absolute() else (self.settings.tuar_root / relative_path)
            if not file_path.exists():
                raise FileNotFoundError(f"Artifact recording file not found: {file_path}")
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
                            for key in ("annotation_source", "split", "channel_names_json")
                            if key in row and row[key] not in {None, ""}
                        },
                    },
                )
            )
        return self.map_labels(RawDatasetBundle.from_recordings(recordings))

    def map_labels(self, bundle: RawDatasetBundle) -> RawDatasetBundle:
        """Map raw artifact labels into class IDs."""
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
                    mapped_label=artifact_label_id(record.raw_label),
                    extra_metadata=dict(record.extra_metadata),
                )
            )
        return RawDatasetBundle.from_recordings(remapped)

    def make_windows(self, bundle: RawDatasetBundle, preprocessor: Any) -> WindowedDataset:
        """Window artifact recordings after preprocessing."""
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
                        "mapped_label": int(record.mapped_label),
                        "sampling_rate": effective_sampling_rate,
                        "start_sample": start,
                        "end_sample": stop,
                        "window_index": index,
                        **record.extra_metadata,
                    }
                )
        if not windows:
            return WindowedDataset(
                windows=np.empty((0, 0, 0), dtype=float),
                labels=np.empty((0,), dtype=int),
                metadata=pd.DataFrame(),
            )
        return WindowedDataset(windows=np.stack(windows, axis=0), labels=np.asarray(labels, dtype=int), metadata=pd.DataFrame(metadata_rows))
