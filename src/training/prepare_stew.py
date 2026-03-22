"""Prepare the STEW workload dataset for downstream decoder training."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import get_settings
from src.datasets.base import RawDatasetBundle, WindowedDataset
from src.datasets.stew_loader import STEWLoader, STEW_CHANNEL_NAMES
from src.features.feature_builder import FeatureBuilder
from src.preprocessing.filters import preprocess_signal
from src.preprocessing.normalization import RecordingRobustScaler, align_signal_channels
from src.preprocessing.splitters import SubjectSplit, train_val_test_subject_split
from src.utils.io import ensure_dir, save_dataframe, save_json, save_json_data, save_pickle
from src.utils.logging_utils import get_logger
from src.utils.seed import set_global_seed

LOGGER = get_logger("training.prepare_stew")


@dataclass(slots=True)
class STEWPreprocessor:
    """Preprocessing bundle for STEW preparation artifacts."""

    channel_names: list[str]
    sampling_rate: float = 128.0
    window_seconds: float = 2.0
    stride_seconds: float = 0.5
    bandpass_low: float | None = 1.0
    bandpass_high: float | None = 40.0
    notch_freq: float | None = None
    normalizer: RecordingRobustScaler = field(default_factory=RecordingRobustScaler)

    def preprocess_only(
        self,
        signal: np.ndarray,
        sampling_rate: float,
        channel_names: list[str] | None,
    ) -> np.ndarray:
        """Align channels and apply deterministic filtering before scaling."""
        aligned = align_signal_channels(signal, channel_names or self.channel_names, self.channel_names)
        if not np.isclose(sampling_rate, self.sampling_rate):
            raise ValueError(f"STEW preparation expects {self.sampling_rate} Hz recordings, received {sampling_rate} Hz.")
        return preprocess_signal(
            aligned,
            sampling_rate=sampling_rate,
            bandpass_low=self.bandpass_low,
            bandpass_high=self.bandpass_high,
            notch_freq=self.notch_freq,
        )

    def fit(self, bundle: RawDatasetBundle) -> "STEWPreprocessor":
        """Fit train-only robust scaling statistics."""
        processed = [
            self.preprocess_only(record.signal, record.sampling_rate, record.channel_names)
            for record in bundle.recordings
        ]
        self.normalizer = RecordingRobustScaler(channel_names=list(self.channel_names)).fit(processed)
        return self

    def transform_raw(
        self,
        signal: np.ndarray,
        sampling_rate: float,
        channel_names: list[str] | None,
    ) -> np.ndarray:
        """Apply the full fitted preprocessing stack to one recording."""
        filtered = self.preprocess_only(signal, sampling_rate, channel_names)
        return self.normalizer.transform(filtered)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare the STEW dataset for workload decoder training.")
    parser.add_argument("--data-root", type=Path, default=None, help="Local STEW dataset root. Defaults to STRESS_DATA_ROOT.")
    parser.add_argument("--artifacts-root", type=Path, default=None, help="Artifact root. Defaults to ARTIFACTS_ROOT.")
    return parser.parse_args()


def _subject_split_frame(split: SubjectSplit) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for name, subjects in (
        ("train", split.train_subjects),
        ("val", split.val_subjects),
        ("test", split.test_subjects),
    ):
        for subject_id in subjects:
            rows.append({"subject_id": subject_id, "split": name})
    return pd.DataFrame(rows)


def _class_balance(values: pd.Series) -> dict[str, int]:
    counts = values.value_counts(dropna=False).sort_index()
    return {str(index): int(count) for index, count in counts.items()}


def _print_sanity_report(recording_metadata: pd.DataFrame, loader: STEWLoader) -> None:
    durations = recording_metadata["duration_seconds"].to_numpy(dtype=float)
    lo_count = int((recording_metadata["condition_raw"] == "lo").sum())
    hi_count = int((recording_metadata["condition_raw"] == "hi").sum())
    LOGGER.info("STEW subjects discovered: %s", recording_metadata["subject_id"].nunique())
    LOGGER.info("STEW files discovered: lo=%s | hi=%s", lo_count, hi_count)
    LOGGER.info(
        "Malformed/skipped files: malformed=%s | skipped=%s",
        len(loader.malformed_files),
        len(loader.skipped_files),
    )
    LOGGER.info(
        "Recording durations (seconds): min=%.2f | max=%.2f | mean=%.2f",
        float(durations.min()),
        float(durations.max()),
        float(durations.mean()),
    )
    LOGGER.info("Binary class balance: %s", _class_balance(recording_metadata["binary_label"]))
    if recording_metadata["workload_rating"].notna().any():
        LOGGER.info("Ordinal class balance: %s", _class_balance(recording_metadata["ordinal_label"]))


def _window_and_featurize(
    loader: STEWLoader,
    bundle: RawDatasetBundle,
    preprocessor: STEWPreprocessor,
    feature_builder: FeatureBuilder,
    split_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    LOGGER.info("Creating %s windows", split_name)
    windowed = loader.make_windows(bundle, preprocessor)
    metadata = windowed.metadata.reset_index(drop=True).copy()
    metadata["split"] = split_name
    if len(windowed.labels) == 0:
        LOGGER.info("No %s windows were produced", split_name)
        metadata["feature_row_index"] = pd.Series(dtype=int)
        return (
            np.empty((0, len(feature_builder.feature_names)), dtype=float),
            np.empty((0,), dtype=int),
            np.empty((0,), dtype=float),
            metadata,
        )

    LOGGER.info("Extracting %s features from %s windows", split_name, len(windowed.labels))
    X, feature_names = feature_builder.build_matrix(
        windowed.windows,
        sampling_rate=preprocessor.sampling_rate,
        channel_names=preprocessor.channel_names,
    )
    if not feature_builder.feature_names:
        feature_builder.feature_names = list(feature_names)
    metadata["feature_row_index"] = np.arange(len(metadata), dtype=int)
    y_binary = windowed.labels.astype(int)
    y_ordinal = pd.to_numeric(metadata["ordinal_target"], errors="coerce").to_numpy(dtype=float)
    return X, y_binary, y_ordinal, metadata


def _summary_payload(
    recording_metadata: pd.DataFrame,
    metadata_windows: pd.DataFrame,
    split_frames: dict[str, pd.DataFrame],
    feature_builder: FeatureBuilder,
    loader: STEWLoader,
    preprocessor: STEWPreprocessor,
) -> dict[str, object]:
    return {
        "number_of_subjects": int(recording_metadata["subject_id"].nunique()),
        "number_of_recordings": int(len(recording_metadata)),
        "number_of_windows": int(len(metadata_windows)),
        "class_balance_per_split": {
            split_name: {
                "binary": _class_balance(frame["binary_label"]) if not frame.empty else {},
                "ordinal": _class_balance(frame["ordinal_label"]) if "ordinal_label" in frame and not frame.empty else {},
            }
            for split_name, frame in split_frames.items()
        },
        "channels_used": list(preprocessor.channel_names),
        "sampling_rate": preprocessor.sampling_rate,
        "window_config": {
            "window_seconds": preprocessor.window_seconds,
            "stride_seconds": preprocessor.stride_seconds,
            "bandpass_low": preprocessor.bandpass_low,
            "bandpass_high": preprocessor.bandpass_high,
            "notch_freq": preprocessor.notch_freq,
        },
        "feature_count": len(feature_builder.feature_names),
        "ratings_available": bool(recording_metadata["workload_rating"].notna().any()),
        "malformed_file_count": len(loader.malformed_files),
        "skipped_file_count": len(loader.skipped_files),
        "missing_file_count": 0,
    }


def main() -> None:
    args = _parse_args()
    settings = get_settings()
    set_global_seed(settings.random_seed)

    data_root = (args.data_root or settings.stress_data_root).expanduser().resolve()
    artifacts_root = ensure_dir((args.artifacts_root or settings.artifacts_root).expanduser().resolve() / "stew")
    LOGGER.info("Preparing STEW from %s", data_root)
    LOGGER.info("Saving STEW artifacts to %s", artifacts_root)

    loader = STEWLoader(data_root=data_root)
    bundle = loader.load_raw()
    recording_metadata = loader.build_metadata().sort_values(["subject_id", "condition_raw"]).reset_index(drop=True)
    _print_sanity_report(recording_metadata, loader)

    split = train_val_test_subject_split(
        bundle,
        train_ratio=settings.split_train,
        val_ratio=settings.split_val,
        test_ratio=settings.split_test,
        seed=settings.random_seed,
    )
    LOGGER.info(
        "Subject split complete: train=%s | val=%s | test=%s",
        len(split.train_subjects),
        len(split.val_subjects),
        len(split.test_subjects),
    )

    preprocessor = STEWPreprocessor(
        channel_names=list(STEW_CHANNEL_NAMES),
        sampling_rate=128.0,
        window_seconds=settings.window_seconds,
        stride_seconds=settings.stride_seconds,
        bandpass_low=settings.bandpass_low,
        bandpass_high=settings.bandpass_high,
        notch_freq=None,
    ).fit(split.train)
    LOGGER.info("Fitted train-only recording normalization on %s train recordings", len(split.train.recordings))

    feature_builder = FeatureBuilder(
        include_absolute_bandpower=True,
        include_log_bandpower=False,
        include_relative_bandpower=True,
        include_temporal_stats=True,
        include_hjorth=True,
        include_ratios=True,
        include_asymmetry=False,
        ratio_feature_names=("alpha_beta_ratio",),
    )

    X_train, y_train_binary, y_train_ordinal, meta_train = _window_and_featurize(
        loader, split.train, preprocessor, feature_builder, "train"
    )
    X_val, y_val_binary, y_val_ordinal, meta_val = _window_and_featurize(
        loader, split.val, preprocessor, feature_builder, "val"
    )
    X_test, y_test_binary, y_test_ordinal, meta_test = _window_and_featurize(
        loader, split.test, preprocessor, feature_builder, "test"
    )
    LOGGER.info("Feature matrix shapes: train=%s | val=%s | test=%s", X_train.shape, X_val.shape, X_test.shape)

    metadata_windows = pd.concat([meta_train, meta_val, meta_test], ignore_index=True)
    subject_split = _subject_split_frame(split)

    np.save(artifacts_root / "X_train.npy", X_train)
    np.save(artifacts_root / "X_val.npy", X_val)
    np.save(artifacts_root / "X_test.npy", X_test)
    np.save(artifacts_root / "y_train_binary.npy", y_train_binary)
    np.save(artifacts_root / "y_val_binary.npy", y_val_binary)
    np.save(artifacts_root / "y_test_binary.npy", y_test_binary)
    np.save(artifacts_root / "y_train_ordinal.npy", y_train_ordinal)
    np.save(artifacts_root / "y_val_ordinal.npy", y_val_ordinal)
    np.save(artifacts_root / "y_test_ordinal.npy", y_test_ordinal)

    save_dataframe(recording_metadata, artifacts_root / "metadata_recordings.csv")
    save_dataframe(metadata_windows, artifacts_root / "metadata_windows.csv")
    save_dataframe(subject_split, artifacts_root / "subject_split.csv")
    save_json_data(list(feature_builder.feature_names), artifacts_root / "feature_names.json")
    save_pickle(preprocessor, artifacts_root / "preprocessing.pkl")
    save_json(
        _summary_payload(
            recording_metadata,
            metadata_windows,
            {"train": meta_train, "val": meta_val, "test": meta_test},
            feature_builder,
            loader,
            preprocessor,
        ),
        artifacts_root / "prep_summary.json",
    )

    print(
        "stew preparation complete\n"
        f"subjects: {recording_metadata['subject_id'].nunique()}\n"
        f"recordings: {len(recording_metadata)}\n"
        f"windows: {len(metadata_windows)}\n"
        f"train features: {X_train.shape}\n"
        f"val features: {X_val.shape}\n"
        f"test features: {X_test.shape}\n"
        f"artifacts: {artifacts_root}"
    )


if __name__ == "__main__":
    main()
