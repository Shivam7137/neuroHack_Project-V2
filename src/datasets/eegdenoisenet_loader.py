"""EEGdenoiseNet-style epoch loader and synthetic artifact augmentation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import Settings, get_settings
from src.datasets.base import RawDatasetBundle, RawRecording
from src.datasets.stress_local_loader import _normalize_signal_shape, _parse_channel_names


@dataclass(slots=True)
class EEGdenoiseNetEpochs:
    """Grouped clean and artifact epochs loaded from disk."""

    clean_epochs: list[np.ndarray]
    eog_epochs: list[np.ndarray]
    emg_epochs: list[np.ndarray]
    channel_names: list[str]
    sampling_rate: float


class EEGdenoiseNetLoader:
    """Load EEGdenoiseNet epochs from a manifest or local folders."""

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
        raise ValueError(f"Unsupported EEGdenoiseNet file format: {path.suffix}")

    def _infer_manifest(self) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        official_files = {
            "clean": self.settings.eegdenoisenet_root / "EEG_all_epochs.npy",
            "eog": self.settings.eegdenoisenet_root / "EOG_all_epochs.npy",
            "emg": self.settings.eegdenoisenet_root / "EMG_all_epochs.npy",
        }
        if any(path.exists() for path in official_files.values()):
            for sample_type, path in official_files.items():
                if path.exists():
                    rows.append(
                        {
                            "file_path": str(path),
                            "sample_type": sample_type,
                            "subject_id": f"{sample_type}_bulk",
                            "bulk_array": True,
                        }
                    )
            return pd.DataFrame(rows)
        for sample_type in ("clean", "eog", "emg"):
            sample_root = self.settings.eegdenoisenet_root / sample_type
            if not sample_root.exists():
                continue
            for path in sorted(sample_root.rglob("*")):
                if not path.is_file() or path.suffix.lower() not in {".npy", ".npz"}:
                    continue
                rows.append(
                    {
                        "file_path": str(path),
                        "sample_type": sample_type,
                        "subject_id": f"{sample_type}_{path.stem}",
                    }
                )
        return pd.DataFrame(rows)

    def load_epochs(self) -> EEGdenoiseNetEpochs | None:
        """Load grouped clean/EOG/EMG epochs from disk when available."""
        manifest_path = self.settings.eegdenoisenet_root / "manifest.csv"
        manifest = pd.read_csv(manifest_path) if manifest_path.exists() else self._infer_manifest()
        if manifest.empty:
            return None
        required = {"file_path", "sample_type"}
        missing = required - set(manifest.columns)
        if missing:
            raise ValueError(f"EEGdenoiseNet manifest is missing required columns: {sorted(missing)}")

        grouped: dict[str, list[np.ndarray]] = {"clean": [], "eog": [], "emg": []}
        channel_names: list[str] = []
        sampling_rate: float | None = None
        for row in manifest.to_dict(orient="records"):
            relative_path = Path(str(row["file_path"]))
            file_path = relative_path if relative_path.is_absolute() else (self.settings.eegdenoisenet_root / relative_path)
            if not file_path.exists():
                raise FileNotFoundError(f"EEGdenoiseNet epoch file not found: {file_path}")
            sample_type = str(row["sample_type"]).strip().lower()
            if sample_type not in grouped:
                raise ValueError("EEGdenoiseNet sample_type must be one of clean/eog/emg.")
            is_bulk_array = bool(row.get("bulk_array"))
            if is_bulk_array:
                bulk_array = np.load(file_path, allow_pickle=True)
                if bulk_array.ndim != 2:
                    raise ValueError(f"Expected bulk EEGdenoiseNet array to be 2D, received shape {bulk_array.shape}.")
                grouped[sample_type].extend(np.asarray(bulk_array[index], dtype=float) for index in range(bulk_array.shape[0]))
                resolved_sampling_rate = float(row["sampling_rate"]) if row.get("sampling_rate") not in {None, ""} else 256.0
                resolved_channels = _parse_channel_names(row.get("channel_names")) or ["Cz"]
            else:
                parsed_channel_names = _parse_channel_names(row.get("channel_names"))
                signal, resolved_sampling_rate, resolved_channels = self._load_signal(
                    file_path,
                    sampling_rate=float(row["sampling_rate"]) if row.get("sampling_rate") not in {None, ""} else None,
                    channel_names=parsed_channel_names,
                )
                grouped[sample_type].append(signal)
            if not channel_names:
                channel_names = resolved_channels
            if sampling_rate is None:
                sampling_rate = resolved_sampling_rate

        return EEGdenoiseNetEpochs(
            clean_epochs=grouped["clean"],
            eog_epochs=grouped["eog"],
            emg_epochs=grouped["emg"],
            channel_names=channel_names,
            sampling_rate=float(sampling_rate or 256.0),
        )


def build_synthetic_artifact_bundle(
    epochs: EEGdenoiseNetEpochs | None,
    *,
    noise_scales: tuple[float, ...] = (0.6, 1.0),
) -> RawDatasetBundle | None:
    """Create clean and synthetic artifact recordings from EEGdenoiseNet epochs."""
    if epochs is None or not epochs.clean_epochs:
        return None

    recordings: list[RawRecording] = []
    def _epoch_signal(epoch: np.ndarray) -> np.ndarray:
        array = np.asarray(epoch, dtype=float)
        return array.reshape(1, -1) if array.ndim == 1 else array

    for clean_index, clean_epoch in enumerate(epochs.clean_epochs):
        subject_id = f"eegdenoise_subject_{clean_index:04d}"
        session_id = f"{subject_id}_clean"
        recordings.append(
            RawRecording(
                signal=_epoch_signal(clean_epoch),
                sampling_rate=epochs.sampling_rate,
                channel_names=list(epochs.channel_names),
                subject_id=subject_id,
                session_id=session_id,
                source_dataset="eegdenoisenet",
                raw_label="clean",
                mapped_label=0,
                extra_metadata={"augmentation_source": "clean"},
            )
        )

        if epochs.eog_epochs:
            eog_epoch = epochs.eog_epochs[clean_index % len(epochs.eog_epochs)]
            for scale_index, scale in enumerate(noise_scales):
                recordings.append(
                    RawRecording(
                        signal=_epoch_signal(np.asarray(clean_epoch, dtype=float) + (scale * np.asarray(eog_epoch, dtype=float))),
                        sampling_rate=epochs.sampling_rate,
                        channel_names=list(epochs.channel_names),
                        subject_id=subject_id,
                        session_id=f"{subject_id}_eyem_{scale_index}",
                        source_dataset="eegdenoisenet_aug",
                        raw_label="eyem",
                        mapped_label=1,
                        extra_metadata={"augmentation_source": "clean_plus_eog", "noise_scale": float(scale)},
                    )
                )

        if epochs.emg_epochs:
            emg_epoch = epochs.emg_epochs[clean_index % len(epochs.emg_epochs)]
            for scale_index, scale in enumerate(noise_scales):
                recordings.append(
                    RawRecording(
                        signal=_epoch_signal(np.asarray(clean_epoch, dtype=float) + (scale * np.asarray(emg_epoch, dtype=float))),
                        sampling_rate=epochs.sampling_rate,
                        channel_names=list(epochs.channel_names),
                        subject_id=subject_id,
                        session_id=f"{subject_id}_musc_{scale_index}",
                        source_dataset="eegdenoisenet_aug",
                        raw_label="musc",
                        mapped_label=5,
                        extra_metadata={"augmentation_source": "clean_plus_emg", "noise_scale": float(scale)},
                    )
                )

    return RawDatasetBundle.from_recordings(recordings)
