"""Artifact-quality training helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import Settings
from src.datasets.artifact_common import ARTIFACT_CLASS_NAMES
from src.datasets.base import RawDatasetBundle, RawRecording
from src.datasets.eegdenoisenet_loader import EEGdenoiseNetLoader, build_synthetic_artifact_bundle
from src.datasets.artifact_tuar_loader import TUARLoader
from src.evaluation.artifact_metrics import artifact_metrics, artifact_per_subject
from src.evaluation.artifact_reports import artifact_predictions_frame, artifact_recording_predictions_frame, artifact_summary_payload
from src.evaluation.plots import save_confusion_matrix_plot, save_distribution_plot
from src.evaluation.reports import comparison_payload
from src.features.feature_builder import FeatureBuilder
from src.models.model_factory import get_model_spec
from src.preprocessing.cleaners import apply_optional_autoreject, apply_optional_pyprep
from src.preprocessing.normalization import DeterministicRecordingNormalizer, FeatureScaler, canonicalize_channel_names, demean_window_channels
from src.preprocessing.quality import WindowQualityConfig, assess_window_quality
from src.preprocessing.splitters import SubjectSplit, train_val_test_subject_split
from src.preprocessing.windowing import create_windows
from src.training.common import PreprocessorBundle, WindowQualitySummary
from src.utils.io import ensure_dir, load_json, save_dataframe, save_json, save_pickle
from src.utils.logging_utils import get_logger

LOGGER = get_logger("training.artifact_pipeline")


@dataclass(slots=True)
class ArtifactPreparedData:
    preprocessor: PreprocessorBundle
    X_train: np.ndarray
    y_train: np.ndarray
    meta_train: pd.DataFrame
    X_val: np.ndarray
    y_val: np.ndarray
    meta_val: pd.DataFrame
    X_test: np.ndarray
    y_test: np.ndarray
    meta_test: pd.DataFrame
    quality_reports: dict[str, dict[str, Any]] = field(default_factory=dict)
    cleaner_reports: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass(slots=True)
class ArtifactTrainingResult:
    model: Any
    model_name: str
    prediction_mode: str
    preprocessor: PreprocessorBundle
    val_metrics: dict[str, Any]
    val_predictions: pd.DataFrame
    test_metrics: dict[str, Any]
    test_predictions: pd.DataFrame


def _combine_bundles(*bundles: RawDatasetBundle | None) -> RawDatasetBundle:
    recordings: list[RawRecording] = []
    for bundle in bundles:
        if bundle is not None:
            recordings.extend(bundle.recordings)
    if not recordings:
        raise FileNotFoundError("No artifact recordings were loaded. Check TUAR_ROOT and EEGDENOISENET_ROOT.")
    return RawDatasetBundle.from_recordings(recordings)


def _build_artifact_preprocessor(train_bundle: RawDatasetBundle, settings: Settings) -> PreprocessorBundle:
    first_record = train_bundle.recordings[0]
    canonical = canonicalize_channel_names(first_record.channel_names)
    return PreprocessorBundle(
        task_name="artifact",
        profile_name="artifact",
        channel_names=canonical,
        raw_normalizer=DeterministicRecordingNormalizer(
            channel_names=canonical,
            dropped_channels=[],
            rereference_mode="none",
            apply_recording_robust_scaling=True,
        ),
        bandpass_low=settings.bandpass_low,
        bandpass_high=settings.bandpass_high,
        notch_freq=settings.notch_freq,
        window_seconds=settings.window_seconds,
        stride_seconds=settings.stride_seconds,
        model_prediction_mode="classifier",
        class_names=list(ARTIFACT_CLASS_NAMES),
        class_to_score={0: 1.0, 1: 0.25, 2: 0.1, 3: 0.05, 4: 0.15, 5: 0.1},
        cleanup_level=settings.cleanup_level,
        apply_window_channel_demean=settings.enable_window_channel_demean,
        quality_config=WindowQualityConfig(
            flat_variance_threshold=settings.flat_variance_threshold,
            max_abs_amplitude_threshold=settings.max_abs_amplitude_threshold,
            max_variance_threshold=settings.max_variance_threshold,
            max_line_noise_ratio=settings.max_line_noise_ratio,
            line_noise_frequency=settings.notch_freq,
        )
        if settings.enable_window_quality_control
        else None,
        feature_group_settings={
            "include_absolute_bandpower": True,
            "include_log_bandpower": True,
            "include_relative_bandpower": True,
            "include_temporal_stats": True,
            "include_hjorth": True,
            "include_ratios": True,
            "include_asymmetry": True,
        },
    )


def _window_to_features(
    windows: np.ndarray,
    metadata: pd.DataFrame,
    feature_builder: FeatureBuilder,
    channel_names: list[str],
) -> tuple[np.ndarray, pd.DataFrame]:
    rows: list[np.ndarray] = []
    for idx, window in enumerate(windows):
        row, _ = feature_builder.build_window(window, sampling_rate=float(metadata.iloc[idx]["sampling_rate"]), channel_names=channel_names)
        rows.append(row)
    if not rows:
        return np.empty((0, len(feature_builder.feature_names)), dtype=float), metadata.iloc[0:0].copy()
    return np.vstack(rows), metadata.reset_index(drop=True).copy()


def _empty_windowed() -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    return np.empty((0, 0, 0), dtype=float), np.empty((0,), dtype=int), pd.DataFrame()


def _window_bundle(
    bundle: RawDatasetBundle,
    preprocessor: PreprocessorBundle,
    settings: Settings,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, dict[str, Any], dict[str, Any]]:
    if not bundle.recordings:
        windows, labels, metadata = _empty_windowed()
        return windows, labels, metadata, WindowQualitySummary().as_dict(), {}

    kept_windows: list[np.ndarray] = []
    kept_labels: list[int] = []
    kept_rows: list[dict[str, Any]] = []
    quality_summary = WindowQualitySummary()
    cleaner_summary = {
        "recordings_processed": 0,
        "pyprep_used_recordings": 0,
        "autoreject_used_recordings": 0,
        "autoreject_rejected_windows": 0,
    }

    for record in bundle.recordings:
        cleaner_summary["recordings_processed"] += 1
        processed = preprocessor.transform_raw(record.signal, record.sampling_rate, record.channel_names)
        pyprep_result = apply_optional_pyprep(
            processed,
            sampling_rate=float(preprocessor.target_sampling_rate or record.sampling_rate),
            channel_names=preprocessor.channel_names,
            enabled=settings.enable_pyprep,
        )
        if pyprep_result.flags.get("pyprep_used"):
            cleaner_summary["pyprep_used_recordings"] += 1
        processed = pyprep_result.signal
        effective_sampling_rate = float(preprocessor.target_sampling_rate or record.sampling_rate)
        recording_windows, bounds = create_windows(
            processed,
            sampling_rate=effective_sampling_rate,
            window_seconds=settings.window_seconds,
            stride_seconds=settings.stride_seconds,
        )
        if recording_windows.size == 0:
            continue

        autoreject_result = apply_optional_autoreject(
            recording_windows,
            sampling_rate=effective_sampling_rate,
            channel_names=preprocessor.channel_names,
            enabled=settings.enable_autoreject,
        )
        if autoreject_result.flags.get("autoreject_used"):
            cleaner_summary["autoreject_used_recordings"] += 1
        cleaner_summary["autoreject_rejected_windows"] += int(autoreject_result.flags.get("autoreject_rejected_windows", 0))
        kept_bounds = [bound for bound, keep in zip(bounds, autoreject_result.keep_mask, strict=True) if keep]
        quality_summary.total_windows += len(kept_bounds)

        for index, (window, (start, stop)) in enumerate(zip(autoreject_result.windows, kept_bounds, strict=True)):
            candidate = demean_window_channels(window) if preprocessor.apply_window_channel_demean else window
            accepted, reasons = (True, [])
            if preprocessor.quality_config is not None:
                accepted, reasons = assess_window_quality(candidate, effective_sampling_rate, preprocessor.quality_config)
            if not accepted:
                quality_summary.rejected_windows += 1
                for reason in reasons:
                    quality_summary.rejection_reasons[reason] = quality_summary.rejection_reasons.get(reason, 0) + 1
                continue

            quality_summary.accepted_windows += 1
            kept_windows.append(candidate)
            kept_labels.append(int(record.mapped_label))
            kept_rows.append(
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
                    "quality_reasons": "|".join(reasons),
                    "pyprep_status": pyprep_result.flags.get("pyprep_status"),
                    "pyprep_bad_channels": "|".join(pyprep_result.flags.get("pyprep_bad_channels", [])),
                    "autoreject_status": autoreject_result.flags.get("autoreject_status"),
                    **record.extra_metadata,
                }
            )

    windows = np.stack(kept_windows, axis=0) if kept_windows else np.empty((0, 0, 0), dtype=float)
    labels = np.asarray(kept_labels, dtype=int)
    metadata = pd.DataFrame(kept_rows)
    return windows, labels, metadata, quality_summary.as_dict(), cleaner_summary


def prepare_artifact_data(settings: Settings) -> tuple[ArtifactPreparedData, SubjectSplit]:
    """Load artifact data, apply preprocessing, and build feature matrices."""
    tuar_loader = TUARLoader(settings=settings)
    tuar_bundle: RawDatasetBundle | None
    try:
        tuar_bundle = tuar_loader.load_raw()
    except FileNotFoundError as exc:
        LOGGER.warning("TUAR data unavailable at %s: %s", settings.tuar_root, exc)
        tuar_bundle = None
    eegdenoise_bundle = build_synthetic_artifact_bundle(EEGdenoiseNetLoader(settings=settings).load_epochs())
    bundle = _combine_bundles(tuar_bundle, eegdenoise_bundle)
    LOGGER.info("Loaded %s artifact recordings across %s subjects", len(bundle.recordings), len({record.subject_id for record in bundle.recordings}))
    split = train_val_test_subject_split(bundle, settings.split_train, settings.split_val, settings.split_test, settings.random_seed)
    preprocessor = _build_artifact_preprocessor(split.train, settings)

    train_windows, train_labels, train_meta, train_qc, train_cleaners = _window_bundle(split.train, preprocessor, settings)
    val_windows, val_labels, val_meta, val_qc, val_cleaners = _window_bundle(split.val, preprocessor, settings)
    test_windows, test_labels, test_meta, test_qc, test_cleaners = _window_bundle(split.test, preprocessor, settings)
    if train_windows.size == 0:
        raise ValueError("Artifact training split produced no valid windows after QC and optional cleaning.")

    feature_builder = FeatureBuilder(**preprocessor.feature_group_settings)
    X_train, meta_train = _window_to_features(train_windows, train_meta, feature_builder, preprocessor.channel_names)
    X_val, meta_val = _window_to_features(val_windows, val_meta, feature_builder, preprocessor.channel_names)
    X_test, meta_test = _window_to_features(test_windows, test_meta, feature_builder, preprocessor.channel_names)
    preprocessor.feature_names = list(feature_builder.feature_names)

    return ArtifactPreparedData(
        preprocessor=preprocessor,
        X_train=X_train,
        y_train=train_labels,
        meta_train=meta_train,
        X_val=X_val,
        y_val=val_labels,
        meta_val=meta_val,
        X_test=X_test,
        y_test=test_labels,
        meta_test=meta_test,
        quality_reports={"train": train_qc, "val": val_qc, "test": test_qc},
        cleaner_reports={"train": train_cleaners, "val": val_cleaners, "test": test_cleaners},
    ), split


def _probabilities(model: Any, features: np.ndarray) -> np.ndarray:
    if not hasattr(model, "predict_proba"):
        raise ValueError("Artifact model must expose predict_proba.")
    return np.asarray(model.predict_proba(features), dtype=float)


def _evaluate_artifact(model: Any, features: np.ndarray, labels: np.ndarray, metadata: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    probabilities = _probabilities(model, features)
    observed_classes = np.asarray(model.classes_, dtype=int)
    expanded = np.zeros((probabilities.shape[0], len(ARTIFACT_CLASS_NAMES)), dtype=float)
    for column_index, class_id in enumerate(observed_classes):
        expanded[:, int(class_id)] = probabilities[:, column_index]
    predicted = np.argmax(expanded, axis=1)
    predictions = artifact_predictions_frame(metadata, labels, predicted, expanded)
    metrics = artifact_metrics(labels, predicted, expanded)
    metrics["per_subject"] = artifact_per_subject(predictions)
    return metrics, predictions


def train_select_and_evaluate_artifact(prepared: ArtifactPreparedData, candidate_names: tuple[str, ...]) -> ArtifactTrainingResult:
    best_payload: tuple[tuple[float, float, float], Any, Any, FeatureScaler | None, dict[str, Any], pd.DataFrame] | None = None
    for candidate_name in candidate_names:
        spec = get_model_spec(candidate_name)
        LOGGER.info("Training artifact candidate model: %s", candidate_name)
        scaler = FeatureScaler().fit(prepared.X_train) if spec.requires_scaling else None
        X_train = scaler.transform(prepared.X_train) if scaler else prepared.X_train
        X_val = scaler.transform(prepared.X_val) if scaler else prepared.X_val
        model = spec.build()
        model.fit(X_train, prepared.y_train)
        val_metrics, val_predictions = _evaluate_artifact(model, X_val, prepared.y_val, prepared.meta_val)
        selection = (
            val_metrics.get("macro_f1") or -1.0,
            val_metrics.get("balanced_accuracy") or -1.0,
            val_metrics.get("binary_f1") or -1.0,
        )
        LOGGER.info("Validation results for %s: %s", candidate_name, selection)
        if best_payload is None or selection > best_payload[0]:
            best_payload = (selection, model, spec, scaler, val_metrics, val_predictions)

    if best_payload is None:
        raise ValueError("No artifact models were trained.")

    _, best_model, best_spec, best_scaler, val_metrics, val_predictions = best_payload
    prepared.preprocessor.feature_scaler = best_scaler
    prepared.preprocessor.model_prediction_mode = best_spec.prediction_mode
    X_test = best_scaler.transform(prepared.X_test) if best_scaler else prepared.X_test
    test_metrics, test_predictions = _evaluate_artifact(best_model, X_test, prepared.y_test, prepared.meta_test)
    val_metrics["quality_control"] = prepared.quality_reports["val"]
    val_metrics["baseline_cleaners"] = prepared.cleaner_reports["val"]
    test_metrics["quality_control"] = prepared.quality_reports["test"]
    test_metrics["baseline_cleaners"] = prepared.cleaner_reports["test"]
    return ArtifactTrainingResult(
        model=best_model,
        model_name=best_spec.name,
        prediction_mode=best_spec.prediction_mode,
        preprocessor=prepared.preprocessor,
        val_metrics=val_metrics,
        val_predictions=val_predictions,
        test_metrics=test_metrics,
        test_predictions=test_predictions,
    )


def save_artifact_artifacts(result: ArtifactTrainingResult, artifacts_root: Path) -> None:
    """Persist artifact model outputs to disk."""
    task_root = ensure_dir(artifacts_root / "artifact")
    plots_root = ensure_dir(task_root / "plots")
    previous_summary = load_json(task_root / "summary.json") if (task_root / "summary.json").exists() else None

    save_pickle({"model": result.model, "model_name": result.model_name, "prediction_mode": result.prediction_mode}, task_root / "model.pkl")
    save_pickle(result.preprocessor, task_root / "preprocessor.pkl")
    save_json(result.val_metrics, task_root / "metrics_val.json")
    save_json(result.test_metrics, task_root / "metrics_test.json")
    save_dataframe(result.val_predictions, task_root / "predictions_val.csv")
    save_dataframe(result.test_predictions, task_root / "predictions_test.csv")

    recording_val_predictions = artifact_recording_predictions_frame(result.val_predictions)
    recording_test_predictions = artifact_recording_predictions_frame(result.test_predictions)
    val_recording_metrics = artifact_metrics(
        recording_val_predictions["true_label"].to_numpy(),
        recording_val_predictions["predicted_label"].to_numpy(),
        recording_val_predictions[[f"probability_{name}" for name in ARTIFACT_CLASS_NAMES]].to_numpy(),
    )
    test_recording_metrics = artifact_metrics(
        recording_test_predictions["true_label"].to_numpy(),
        recording_test_predictions["predicted_label"].to_numpy(),
        recording_test_predictions[[f"probability_{name}" for name in ARTIFACT_CLASS_NAMES]].to_numpy(),
    )
    val_recording_metrics["per_subject"] = artifact_per_subject(recording_val_predictions)
    test_recording_metrics["per_subject"] = artifact_per_subject(recording_test_predictions)

    save_dataframe(recording_val_predictions, task_root / "predictions_recording_val.csv")
    save_dataframe(recording_test_predictions, task_root / "predictions_recording_test.csv")
    save_json(val_recording_metrics, task_root / "metrics_recording_val.json")
    save_json(test_recording_metrics, task_root / "metrics_recording_test.json")

    summary = artifact_summary_payload(result.model_name, result.val_metrics, result.test_metrics)
    summary["validation_recording"] = val_recording_metrics
    summary["test_recording"] = test_recording_metrics
    save_json(summary, task_root / "summary.json")
    if previous_summary is not None:
        save_json(comparison_payload(previous_summary, summary), task_root / "comparison.json")

    save_confusion_matrix_plot(result.test_metrics["confusion_matrix"], list(ARTIFACT_CLASS_NAMES), plots_root / "confusion_matrix.png", "Artifact Confusion Matrix")
    save_confusion_matrix_plot(result.test_metrics["confusion_matrix"], list(ARTIFACT_CLASS_NAMES), task_root / "confusion_matrix.png", "Artifact Confusion Matrix")
    save_distribution_plot(result.test_predictions["quality_score"].to_numpy(), plots_root / "quality_distribution.png", "Artifact Quality Distribution", "Quality score")
    save_distribution_plot(result.test_predictions["quality_score"].to_numpy(), task_root / "quality_distribution.png", "Artifact Quality Distribution", "Quality score")


def artifact_terminal_summary(result: ArtifactTrainingResult, split: SubjectSplit) -> str:
    """Return a concise terminal summary for artifact training."""
    return (
        "artifact complete\n"
        f"model: {result.model_name}\n"
        f"train subjects: {len(split.train_subjects)} | val subjects: {len(split.val_subjects)} | test subjects: {len(split.test_subjects)}\n"
        f"validation macro_f1: {result.val_metrics.get('macro_f1')}\n"
        f"test macro_f1: {result.test_metrics.get('macro_f1')}\n"
        f"test binary_f1: {result.test_metrics.get('binary_f1')}\n"
        f"test metrics: {result.test_metrics}"
    )
