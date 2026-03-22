"""Loader for the PhysioNet EEGMAT dataset."""

from __future__ import annotations

import shutil
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import Settings, get_settings
from src.datasets.base import BaseDatasetLoader, RawDatasetBundle, RawRecording, WindowedDataset
from src.preprocessing.windowing import create_windows

EEGMAT_ZIP_URL = "https://physionet.org/static/published-projects/eegmat/eeg-during-mental-arithmetic-tasks-1.0.0.zip"
LOCAL_EEGMAT_SUFFIXES = (".edf", ".npz", ".npy")


def _read_edf_file(path: Path) -> tuple[np.ndarray, float, list[str]]:
    """Read one EDF file via optional backends."""
    try:
        import pyedflib  # type: ignore

        reader = pyedflib.EdfReader(str(path))
        try:
            channel_names = reader.getSignalLabels()
            sampling_rates = reader.getSampleFrequencies()
            signal = np.vstack([reader.readSignal(idx) for idx in range(reader.signals_in_file)]).astype(float)
            sampling_rate = float(sampling_rates[0])
            return signal, sampling_rate, list(channel_names)
        finally:
            reader.close()
    except ImportError:
        pass

    try:
        import mne  # type: ignore

        raw = mne.io.read_raw_edf(str(path), preload=True, verbose="ERROR")
        signal = raw.get_data().astype(float)
        sampling_rate = float(raw.info["sfreq"])
        channel_names = list(raw.ch_names)
        return signal, sampling_rate, channel_names
    except ImportError as exc:
        raise RuntimeError(_edf_support_error(str(path))) from exc


def _edf_support_error(path: str) -> str:
    """Return a concrete fallback message for missing EDF support."""
    return (
        "EDF support is not installed, so EEGMAT cannot be read automatically.\n"
        f"Tried to load: {path}\n"
        "Install one of the optional readers and retry:\n"
        "  python -m pip install pyedflib\n"
        "or\n"
        "  python -m pip install mne\n"
        "If you prefer local-only access, place the extracted EEGMAT files under EEGMAT_ROOT.\n"
        "Expected files include Subject00_1.edf, Subject00_2.edf, and subject-info.csv."
    )


def _normalize_signal_shape(signal: np.ndarray) -> np.ndarray:
    """Normalize local fallback arrays to [channels, samples]."""
    if signal.ndim != 2:
        raise ValueError(f"Expected a 2D EEG array, received shape {signal.shape}.")
    return signal.astype(float) if signal.shape[0] <= signal.shape[1] else signal.T.astype(float)


class EEGMATLoader(BaseDatasetLoader):
    """Programmatic or local-folder loader for EEGMAT."""

    dataset_name = "eegmat"

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def _maybe_download(self) -> None:
        root = self.settings.eegmat_root
        subject_info = root / "subject-info.csv"
        local_files = any(root.glob("Subject*_*.edf")) or any(root.glob("Subject*_*.npz")) or any(root.glob("Subject*_*.npy"))
        if subject_info.exists() or local_files:
            return
        if not self.settings.auto_download_eegmat:
            raise FileNotFoundError(
                f"EEGMAT dataset not found at {root}. "
                "Enable AUTO_DOWNLOAD_EEGMAT=true or place the extracted files there."
            )
        root.mkdir(parents=True, exist_ok=True)
        archive_path = root / "eegmat.zip"
        try:
            urllib.request.urlretrieve(EEGMAT_ZIP_URL, archive_path)
            with zipfile.ZipFile(archive_path, "r") as archive:
                archive.extractall(root)
            extracted_root = root / "eeg-during-mental-arithmetic-tasks-1.0.0"
            if extracted_root.exists():
                for child in extracted_root.iterdir():
                    destination = root / child.name
                    if destination.exists():
                        if destination.is_dir():
                            shutil.rmtree(destination)
                        else:
                            destination.unlink()
                    shutil.move(str(child), str(destination))
                shutil.rmtree(extracted_root, ignore_errors=True)
        except (urllib.error.URLError, zipfile.BadZipFile) as exc:
            raise RuntimeError(
                "Automatic EEGMAT download failed. "
                f"Place the extracted PhysioNet dataset under {root} and retry. "
                f"Underlying error: {exc}"
            ) from exc
        finally:
            if archive_path.exists():
                archive_path.unlink()

    def load_raw(self) -> RawDatasetBundle:
        """Load raw EEGMAT EDF recordings."""
        self._maybe_download()
        subject_info_path = self.settings.eegmat_root / "subject-info.csv"
        subject_info = pd.read_csv(subject_info_path) if subject_info_path.exists() else pd.DataFrame()
        subject_lookup = {
            str(row["Subject"]): row
            for _, row in subject_info.iterrows()
        }
        recordings: list[RawRecording] = []
        paths = []
        for suffix in LOCAL_EEGMAT_SUFFIXES:
            paths.extend(sorted(self.settings.eegmat_root.glob(f"Subject*_*{suffix}")))
        for path in paths:
            if path.suffix.lower() == ".edf":
                signal, sampling_rate, channel_names = _read_edf_file(path)
            elif path.suffix.lower() == ".npz":
                payload = np.load(path, allow_pickle=True)
                signal_key = "signal" if "signal" in payload else list(payload.keys())[0]
                signal = _normalize_signal_shape(np.asarray(payload[signal_key], dtype=float))
                sampling_rate = float(payload["sampling_rate"]) if "sampling_rate" in payload else 128.0
                if "channel_names" in payload:
                    channel_names = [str(item) for item in payload["channel_names"].tolist()]
                else:
                    channel_names = [f"ch_{idx:02d}" for idx in range(signal.shape[0])]
            else:
                signal = _normalize_signal_shape(np.load(path).astype(float))
                sampling_rate = 128.0
                channel_names = [f"ch_{idx:02d}" for idx in range(signal.shape[0])]
            stem = path.stem
            subject_part, session_part = stem.split("_", maxsplit=1)
            raw_label = "baseline" if session_part == "1" else "mental_arithmetic"
            info_row = subject_lookup.get(subject_part)
            recordings.append(
                RawRecording(
                    signal=signal,
                    sampling_rate=sampling_rate,
                    channel_names=channel_names,
                    subject_id=subject_part,
                    session_id=session_part,
                    source_dataset=self.dataset_name,
                    raw_label=raw_label,
                    mapped_label=None,
                    extra_metadata={
                        "file_path": str(path),
                        **(
                            {
                                "age": int(info_row["Age"]),
                                "gender": str(info_row["Gender"]),
                                "recording_year": int(info_row["Recording year"]),
                                "number_of_subtractions": float(info_row["Number of subtractions"]),
                                "count_quality": int(info_row["Count quality"]),
                            }
                            if info_row is not None
                            else {}
                        ),
                    },
                )
            )
        if not recordings:
            raise FileNotFoundError(
                f"No EEGMAT EDF files were found under {self.settings.eegmat_root}. "
                "Expected files like Subject00_1.edf and Subject00_2.edf. "
                "For local development, SubjectXX_1.npz and SubjectXX_2.npz are also supported."
            )
        return self.map_labels(RawDatasetBundle.from_recordings(recordings))

    def map_labels(self, bundle: RawDatasetBundle) -> RawDatasetBundle:
        """Map EEGMAT labels into a binary concentration target."""
        remapped: list[RawRecording] = []
        for record in bundle.recordings:
            mapped_label = 0 if record.raw_label in {"baseline", "rest", "neutral"} else 1
            remapped.append(
                RawRecording(
                    signal=record.signal,
                    sampling_rate=record.sampling_rate,
                    channel_names=record.channel_names,
                    subject_id=record.subject_id,
                    session_id=record.session_id,
                    source_dataset=record.source_dataset,
                    raw_label=record.raw_label,
                    mapped_label=mapped_label,
                    extra_metadata=dict(record.extra_metadata),
                )
            )
        return RawDatasetBundle.from_recordings(remapped)

    def make_windows(self, bundle: RawDatasetBundle, preprocessor: Any) -> WindowedDataset:
        """Window EEGMAT recordings with a fitted preprocessor."""
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
