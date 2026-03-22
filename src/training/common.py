"""Shared training and evaluation pipeline helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import resample_poly
from sklearn.metrics import balanced_accuracy_score, f1_score

from src.config import Settings
from src.datasets.base import BaseDatasetLoader, RawDatasetBundle, WindowedDataset
from src.evaluation.metrics import concentration_metrics, concentration_per_subject, pairwise_ranking_metrics, stress_metrics, stress_per_subject
from src.evaluation.plots import save_confusion_matrix_plot, save_distribution_plot, save_roc_curve_plot, save_score_scatter_plot
from src.evaluation.reports import (
    comparison_payload,
    concentration_pairwise_frame,
    concentration_predictions_frame,
    concentration_recording_predictions_frame,
    stress_pairwise_frame,
    stress_predictions_frame,
    stress_recording_predictions_frame,
    summary_payload,
)
from src.features.feature_builder import FeatureBuilder
from src.models.model_factory import get_model_spec
from src.models.stress_model import STRESS_CLASS_NAMES, STRESS_CLASS_TO_SCORE
from src.preprocessing.filters import preprocess_signal
from src.preprocessing.normalization import DeterministicRecordingNormalizer, FeatureScaler, canonicalize_channel_names, demean_window_channels
from src.preprocessing.quality import WindowQualityConfig, assess_window_quality
from src.preprocessing.splitters import SubjectSplit, train_val_test_subject_split
from src.utils.io import ensure_dir, load_json, save_dataframe, save_json, save_pickle
from src.utils.logging_utils import get_logger

LOGGER = get_logger("training.common")


def _concentration_scores() -> dict[int, float]:
    return {0: 0.0, 1: 1.0}


def _score_to_stress_class(score: float) -> int:
    values = np.asarray([STRESS_CLASS_TO_SCORE[idx] for idx in range(len(STRESS_CLASS_NAMES))], dtype=float)
    return int(np.argmin(np.abs(values - score)))


def _score_to_rating(score: float) -> int:
    return int(np.clip(np.rint(score * 8.0 + 1.0), 1, 9))


def _rating_to_three_level(rating: int) -> str:
    if rating <= 3:
        return "low"
    if rating <= 6:
        return "moderate"
    return "high"


@dataclass(slots=True)
class WindowQualitySummary:
    total_windows: int = 0
    accepted_windows: int = 0
    rejected_windows: int = 0
    rejection_reasons: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "total_windows": self.total_windows,
            "accepted_windows": self.accepted_windows,
            "rejected_windows": self.rejected_windows,
            "rejection_rate": self.rejected_windows / self.total_windows if self.total_windows else 0.0,
            "rejection_reasons": dict(sorted(self.rejection_reasons.items())),
        }


@dataclass(slots=True)
class PreprocessorBundle:
    task_name: str
    profile_name: str
    channel_names: list[str]
    raw_normalizer: DeterministicRecordingNormalizer
    feature_names: list[str] = field(default_factory=list)
    feature_scaler: FeatureScaler | None = None
    bandpass_low: float | None = None
    bandpass_high: float | None = None
    notch_freq: float | None = None
    window_seconds: float = 2.0
    stride_seconds: float = 0.5
    model_prediction_mode: str = "classifier"
    class_names: list[str] = field(default_factory=list)
    class_to_score: dict[int, float] = field(default_factory=dict)
    dropped_channels: list[str] = field(default_factory=list)
    rereference_mode: str = "none"
    target_sampling_rate: float | None = None
    trim_seconds_start: float = 0.0
    trim_seconds_end: float = 0.0
    cleanup_level: str = "none"
    apply_window_channel_demean: bool = False
    quality_config: WindowQualityConfig | None = None
    decision_threshold: float | None = None
    feature_group_settings: dict[str, bool] = field(default_factory=dict)

    def _apply_channel_policy(self, signal: np.ndarray, channel_names: list[str] | None) -> np.ndarray:
        current = canonicalize_channel_names(channel_names or self.channel_names)
        if signal.shape[0] != len(current):
            raise ValueError("Signal channel count does not match provided channel names.")
        index_map = {name: idx for idx, name in enumerate(current)}
        missing = [name for name in self.channel_names if name not in index_map]
        if missing:
            raise ValueError(f"Missing required channels: {missing}")
        return signal[[index_map[name] for name in self.channel_names], :]

    def _resample(self, signal: np.ndarray, sampling_rate: float) -> tuple[np.ndarray, float]:
        if self.target_sampling_rate is None or np.isclose(self.target_sampling_rate, sampling_rate):
            return signal, sampling_rate
        return resample_poly(signal, up=int(round(self.target_sampling_rate)), down=int(round(sampling_rate)), axis=-1), float(self.target_sampling_rate)

    def _trim(self, signal: np.ndarray, sampling_rate: float) -> np.ndarray:
        start = int(round(self.trim_seconds_start * sampling_rate))
        end = int(round(self.trim_seconds_end * sampling_rate))
        stop = signal.shape[-1] - end if end else signal.shape[-1]
        if stop <= start:
            raise ValueError("Trim settings removed the entire recording.")
        return signal[:, start:stop]

    def transform_raw_with_sampling_rate(
        self,
        signal: np.ndarray,
        sampling_rate: float,
        channel_names: list[str] | None,
        *,
        apply_trim: bool = True,
    ) -> tuple[np.ndarray, float]:
        processed = self._apply_channel_policy(signal, channel_names)
        processed = preprocess_signal(
            processed,
            sampling_rate=sampling_rate,
            bandpass_low=self.bandpass_low,
            bandpass_high=self.bandpass_high,
            notch_freq=self.notch_freq,
            cleanup_level=self.cleanup_level,
        )
        processed, sampling_rate = self._resample(processed, sampling_rate)
        if apply_trim:
            processed = self._trim(processed, sampling_rate)
        return self.raw_normalizer.transform(processed, self.channel_names), float(sampling_rate)

    def transform_raw(
        self,
        signal: np.ndarray,
        sampling_rate: float,
        channel_names: list[str] | None,
        *,
        apply_trim: bool = True,
    ) -> np.ndarray:
        processed, _ = self.transform_raw_with_sampling_rate(
            signal,
            sampling_rate,
            channel_names,
            apply_trim=apply_trim,
        )
        return processed

    def transform_features(self, features: np.ndarray) -> np.ndarray:
        return self.feature_scaler.transform(features) if self.feature_scaler else features


@dataclass(slots=True)
class PreparedData:
    preprocessor: PreprocessorBundle
    X_train: np.ndarray
    y_train: np.ndarray
    target_scores_train: np.ndarray
    meta_train: pd.DataFrame
    X_val: np.ndarray
    y_val: np.ndarray
    target_scores_val: np.ndarray
    meta_val: pd.DataFrame
    X_test: np.ndarray
    y_test: np.ndarray
    target_scores_test: np.ndarray
    meta_test: pd.DataFrame
    quality_reports: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass(slots=True)
class TrainingResult:
    model: Any
    model_name: str
    prediction_mode: str
    preprocessor: PreprocessorBundle
    decision_threshold: float | None
    val_metrics: dict[str, Any]
    val_predictions: pd.DataFrame
    test_metrics: dict[str, Any]
    test_predictions: pd.DataFrame


def _expand_classifier_probabilities(probabilities: np.ndarray, observed_classes: np.ndarray, n_classes: int) -> np.ndarray:
    expanded = np.zeros((probabilities.shape[0], n_classes), dtype=float)
    for column_index, class_id in enumerate(observed_classes):
        expanded[:, int(class_id)] = probabilities[:, column_index]
    return expanded


def _determine_profile(task_name: str, train_bundle: RawDatasetBundle, settings: Settings) -> dict[str, Any]:
    first_record = train_bundle.recordings[0]
    canonical = canonicalize_channel_names(first_record.channel_names)
    if task_name == "concentration":
        channel_names = [name for name in canonical if name not in {"A2-A1", "ECG"}]
        return {
            "profile_name": "eegmat",
            "channel_names": channel_names,
            "dropped_channels": [name for name in canonical if name not in channel_names],
            "rereference_mode": "none",
            "target_sampling_rate": settings.eegmat_target_sampling_rate,
            "trim_seconds_start": 0.0,
            "trim_seconds_end": 0.0,
        }
    source_variant = first_record.extra_metadata.get("source_variant")
    return {
        "profile_name": "stew" if source_variant == "stew" else "stress_local",
        "channel_names": canonical,
        "dropped_channels": [],
        "rereference_mode": "average" if source_variant == "stew" else "none",
        "target_sampling_rate": None,
        "trim_seconds_start": settings.stew_trim_seconds if source_variant == "stew" else 0.0,
        "trim_seconds_end": settings.stew_trim_seconds if source_variant == "stew" else 0.0,
    }


def _build_preprocessor(task_name: str, train_bundle: RawDatasetBundle, settings: Settings) -> PreprocessorBundle:
    profile = _determine_profile(task_name, train_bundle, settings)
    return PreprocessorBundle(
        task_name=task_name,
        profile_name=profile["profile_name"],
        channel_names=list(profile["channel_names"]),
        raw_normalizer=DeterministicRecordingNormalizer(
            channel_names=list(profile["channel_names"]),
            dropped_channels=list(profile["dropped_channels"]),
            rereference_mode=profile["rereference_mode"],
            apply_recording_robust_scaling=True,
        ),
        bandpass_low=settings.bandpass_low,
        bandpass_high=settings.bandpass_high,
        notch_freq=settings.notch_freq,
        window_seconds=settings.window_seconds,
        stride_seconds=settings.stride_seconds,
        class_names=["rest", "concentration"] if task_name == "concentration" else list(STRESS_CLASS_NAMES),
        class_to_score=_concentration_scores() if task_name == "concentration" else dict(STRESS_CLASS_TO_SCORE),
        dropped_channels=list(profile["dropped_channels"]),
        rereference_mode=profile["rereference_mode"],
        target_sampling_rate=profile["target_sampling_rate"],
        trim_seconds_start=profile["trim_seconds_start"],
        trim_seconds_end=profile["trim_seconds_end"],
        cleanup_level=settings.cleanup_level,
        apply_window_channel_demean=settings.enable_window_channel_demean,
        quality_config=WindowQualityConfig(
            flat_variance_threshold=settings.flat_variance_threshold,
            max_abs_amplitude_threshold=settings.max_abs_amplitude_threshold,
            max_variance_threshold=settings.max_variance_threshold,
            max_line_noise_ratio=settings.max_line_noise_ratio,
            line_noise_frequency=settings.notch_freq,
        ) if settings.enable_window_quality_control else None,
        feature_group_settings={
            "include_log_bandpower": True,
            "include_relative_bandpower": True,
            "include_temporal_stats": True,
            "include_hjorth": True,
            "include_ratios": True,
            "include_asymmetry": True,
        },
    )


def _prepare_target_scores(task_name: str, metadata: pd.DataFrame, labels: np.ndarray) -> np.ndarray:
    if task_name == "stress":
        if "target_score" in metadata.columns:
            fallback = np.asarray([STRESS_CLASS_TO_SCORE[int(label)] for label in labels], dtype=float)
            target = pd.to_numeric(metadata["target_score"], errors="coerce").to_numpy(dtype=float)
            return np.where(np.isnan(target), fallback, target)
        return np.asarray([STRESS_CLASS_TO_SCORE[int(label)] for label in labels], dtype=float)
    return labels.astype(float)


def _filter_windows(windowed: WindowedDataset, preprocessor: PreprocessorBundle) -> tuple[WindowedDataset, WindowQualitySummary]:
    summary = WindowQualitySummary(total_windows=int(len(windowed.labels)))
    if len(windowed.labels) == 0:
        return windowed, summary
    kept_windows: list[np.ndarray] = []
    kept_labels: list[int] = []
    kept_rows: list[dict[str, Any]] = []
    for window, label, (_, row) in zip(windowed.windows, windowed.labels, windowed.metadata.iterrows(), strict=True):
        candidate = demean_window_channels(window) if preprocessor.apply_window_channel_demean else window
        accepted, reasons = (True, [])
        if preprocessor.quality_config is not None:
            accepted, reasons = assess_window_quality(candidate, float(row["sampling_rate"]), preprocessor.quality_config)
        if accepted:
            kept_windows.append(candidate)
            kept_labels.append(int(label))
            kept_rows.append(dict(row))
            summary.accepted_windows += 1
        else:
            summary.rejected_windows += 1
            for reason in reasons:
                summary.rejection_reasons[reason] = summary.rejection_reasons.get(reason, 0) + 1
    filtered = WindowedDataset(
        windows=np.stack(kept_windows, axis=0) if kept_windows else np.empty((0, 0, 0), dtype=float),
        labels=np.asarray(kept_labels, dtype=int),
        metadata=pd.DataFrame(kept_rows),
    )
    return filtered, summary


def _window_to_features(windows: np.ndarray, metadata: pd.DataFrame, feature_builder: FeatureBuilder, channel_names: list[str]) -> tuple[np.ndarray, pd.DataFrame]:
    rows: list[np.ndarray] = []
    for idx, window in enumerate(windows):
        row, _ = feature_builder.build_window(window, sampling_rate=float(metadata.iloc[idx]["sampling_rate"]), channel_names=channel_names)
        rows.append(row)
    if not rows:
        return np.empty((0, len(feature_builder.feature_names)), dtype=float), metadata.iloc[0:0].copy()
    return np.vstack(rows), metadata.reset_index(drop=True).copy()


def prepare_data(loader: BaseDatasetLoader, task_name: str, settings: Settings) -> tuple[PreparedData, SubjectSplit]:
    root_path = getattr(loader, "settings", settings).eegmat_root if task_name == "concentration" else getattr(loader, "settings", settings).stress_data_root
    LOGGER.info("Loading %s dataset from %s", task_name, root_path)
    bundle = loader.load_raw()
    LOGGER.info("Loaded %s recordings across %s subjects", len(bundle.recordings), len({record.subject_id for record in bundle.recordings}))
    split = train_val_test_subject_split(bundle, settings.split_train, settings.split_val, settings.split_test, settings.random_seed)
    LOGGER.info("Subject split complete: train=%s, val=%s, test=%s", len(split.train_subjects), len(split.val_subjects), len(split.test_subjects))
    preprocessor = _build_preprocessor(task_name, split.train, settings)
    LOGGER.info("Using preprocessing profile=%s | channels=%s | dropped=%s | reref=%s | target_fs=%s", preprocessor.profile_name, len(preprocessor.channel_names), preprocessor.dropped_channels, preprocessor.rereference_mode, preprocessor.target_sampling_rate)

    train_windowed, train_qc = _filter_windows(loader.make_windows(split.train, preprocessor), preprocessor)
    val_windowed, val_qc = _filter_windows(loader.make_windows(split.val, preprocessor), preprocessor)
    test_windowed, test_qc = _filter_windows(loader.make_windows(split.test, preprocessor), preprocessor)
    LOGGER.info("Window counts after QC: train=%s, val=%s, test=%s", len(train_windowed.labels), len(val_windowed.labels), len(test_windowed.labels))
    if train_windowed.windows.size == 0:
        raise ValueError("Training split produced no valid EEG windows after quality control.")

    feature_builder = FeatureBuilder(**preprocessor.feature_group_settings)
    X_train, meta_train = _window_to_features(train_windowed.windows, train_windowed.metadata, feature_builder, preprocessor.channel_names)
    X_val, meta_val = _window_to_features(val_windowed.windows, val_windowed.metadata, feature_builder, preprocessor.channel_names)
    X_test, meta_test = _window_to_features(test_windowed.windows, test_windowed.metadata, feature_builder, preprocessor.channel_names)
    preprocessor.feature_names = list(feature_builder.feature_names)
    LOGGER.info("Feature matrices ready: train=%s, val=%s, test=%s, features=%s", X_train.shape, X_val.shape, X_test.shape, len(preprocessor.feature_names))
    return PreparedData(
        preprocessor=preprocessor,
        X_train=X_train,
        y_train=train_windowed.labels,
        target_scores_train=_prepare_target_scores(task_name, meta_train, train_windowed.labels),
        meta_train=meta_train,
        X_val=X_val,
        y_val=val_windowed.labels,
        target_scores_val=_prepare_target_scores(task_name, meta_val, val_windowed.labels),
        meta_val=meta_val,
        X_test=X_test,
        y_test=test_windowed.labels,
        target_scores_test=_prepare_target_scores(task_name, meta_test, test_windowed.labels),
        meta_test=meta_test,
        quality_reports={"train": train_qc.as_dict(), "val": val_qc.as_dict(), "test": test_qc.as_dict()},
    ), split


def _probabilities_or_none(model: Any, features: np.ndarray) -> np.ndarray | None:
    return model.predict_proba(features) if hasattr(model, "predict_proba") else None


def _select_concentration_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    best = (float("-inf"), float("-inf"), 0.5)
    for threshold in np.linspace(0.1, 0.9, 33):
        predicted = (probabilities >= threshold).astype(int)
        candidate = (
            balanced_accuracy_score(y_true, predicted),
            f1_score(y_true, predicted, zero_division=0),
            float(threshold),
        )
        if candidate > best:
            best = candidate
    return float(best[2])


def _evaluate_concentration(model: Any, features: np.ndarray, labels: np.ndarray, metadata: pd.DataFrame, threshold: float) -> tuple[dict[str, Any], pd.DataFrame]:
    probabilities = _probabilities_or_none(model, features)
    if probabilities is None:
        raise ValueError("Concentration model must expose predict_proba.")
    positive_probabilities = probabilities[:, 1]
    predicted = (positive_probabilities >= threshold).astype(int)
    predictions = concentration_predictions_frame(metadata, labels, predicted, positive_probabilities, threshold)
    metrics = concentration_metrics(labels, predicted, positive_probabilities)
    metrics["threshold"] = threshold
    metrics["per_subject"] = concentration_per_subject(predictions)
    return metrics, predictions


def _evaluate_stress_classifier(model: Any, features: np.ndarray, labels: np.ndarray, target_scores: np.ndarray, metadata: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    probabilities = _probabilities_or_none(model, features)
    if probabilities is None:
        raise ValueError("Stress classifier must expose predict_proba.")
    probabilities = _expand_classifier_probabilities(probabilities, np.asarray(model.classes_), len(STRESS_CLASS_NAMES))
    expected_scores = np.sum(probabilities * np.asarray([STRESS_CLASS_TO_SCORE[idx] for idx in range(len(STRESS_CLASS_NAMES))], dtype=float), axis=1)
    predicted = np.argmax(probabilities, axis=1)
    predictions = stress_predictions_frame(
        metadata,
        labels,
        predicted,
        probabilities,
        class_names=list(STRESS_CLASS_NAMES),
        class_to_score=dict(STRESS_CLASS_TO_SCORE),
        predicted_scores=expected_scores,
        true_scores=target_scores,
    )
    metrics = stress_metrics(labels, predicted, target_scores, expected_scores, ratings=metadata["rating"].to_numpy() if "rating" in metadata.columns else None)
    metrics["per_subject"] = stress_per_subject(predictions)
    return metrics, predictions


def _evaluate_stress_regressor(model: Any, features: np.ndarray, labels: np.ndarray, target_scores: np.ndarray, metadata: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    predicted_scores = np.clip(model.predict(features), 0.0, 1.0)
    predicted = np.asarray([_score_to_stress_class(score) for score in predicted_scores], dtype=int)
    predictions = stress_predictions_frame(
        metadata,
        labels,
        predicted,
        probabilities=None,
        class_names=list(STRESS_CLASS_NAMES),
        class_to_score=dict(STRESS_CLASS_TO_SCORE),
        predicted_scores=predicted_scores,
        true_scores=target_scores,
    )
    metrics = stress_metrics(labels, predicted, target_scores, predicted_scores, ratings=metadata["rating"].to_numpy() if "rating" in metadata.columns else None)
    metrics["per_subject"] = stress_per_subject(predictions)
    return metrics, predictions


def _recording_and_pairwise_outputs(task_name: str, predictions: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame, dict[str, Any]]:
    if task_name == "concentration":
        recording_predictions = concentration_recording_predictions_frame(predictions)
        recording_metrics = concentration_metrics(
            recording_predictions["true_label"].to_numpy(),
            recording_predictions["predicted_label"].to_numpy(),
            recording_predictions["concentration_probability"].to_numpy(),
        )
        recording_metrics["per_subject"] = concentration_per_subject(recording_predictions)
        pairwise_predictions = concentration_pairwise_frame(recording_predictions)
    else:
        recording_predictions = stress_recording_predictions_frame(
            predictions,
            class_names=list(STRESS_CLASS_NAMES),
            class_to_score=dict(STRESS_CLASS_TO_SCORE),
        )
        ratings = recording_predictions["rating"].to_numpy() if "rating" in recording_predictions.columns else None
        recording_metrics = stress_metrics(
            recording_predictions["true_label"].to_numpy(),
            recording_predictions["predicted_label"].to_numpy(),
            recording_predictions["true_score"].to_numpy(),
            recording_predictions["predicted_score"].to_numpy(),
            ratings=ratings,
        )
        recording_metrics["per_subject"] = stress_per_subject(recording_predictions)
        pairwise_predictions = stress_pairwise_frame(recording_predictions)
    pairwise_metrics = pairwise_ranking_metrics(pairwise_predictions)
    return recording_predictions, recording_metrics, pairwise_predictions, pairwise_metrics


def _selection_value(task_name: str, prediction_mode: str, metrics: dict[str, Any]) -> tuple[float, float, float]:
    if task_name == "concentration":
        return (metrics.get("roc_auc") or -1.0, metrics.get("balanced_accuracy") or -1.0, metrics.get("f1") or -1.0)
    if prediction_mode == "regressor":
        return (metrics.get("spearman") or -1.0, -(metrics.get("mae") or 1e9), metrics.get("quadratic_weighted_kappa") or -1.0)
    return (metrics.get("macro_f1") or -1.0, -(metrics.get("mae") or 1e9), metrics.get("spearman") or -1.0)


def train_select_and_evaluate(prepared: PreparedData, candidate_names: tuple[str, ...], task_name: str) -> TrainingResult:
    best_payload: tuple[tuple[float, float, float], Any, Any, FeatureScaler | None, float | None, dict[str, Any], pd.DataFrame] | None = None
    for candidate_name in candidate_names:
        spec = get_model_spec(candidate_name)
        LOGGER.info("Training candidate model: %s", candidate_name)
        scaler = FeatureScaler().fit(prepared.X_train) if spec.requires_scaling else None
        X_train = scaler.transform(prepared.X_train) if scaler else prepared.X_train
        X_val = scaler.transform(prepared.X_val) if scaler else prepared.X_val
        model = spec.build()
        target = prepared.target_scores_train if spec.prediction_mode == "regressor" and task_name == "stress" else prepared.y_train
        model.fit(X_train, target)

        threshold: float | None = None
        if task_name == "concentration":
            val_probabilities = _probabilities_or_none(model, X_val)
            if val_probabilities is None:
                raise ValueError("Concentration model must expose predict_proba.")
            threshold = _select_concentration_threshold(prepared.y_val, val_probabilities[:, 1])
            val_metrics, val_predictions = _evaluate_concentration(model, X_val, prepared.y_val, prepared.meta_val, threshold)
        elif spec.prediction_mode == "regressor":
            val_metrics, val_predictions = _evaluate_stress_regressor(model, X_val, prepared.y_val, prepared.target_scores_val, prepared.meta_val)
        else:
            val_metrics, val_predictions = _evaluate_stress_classifier(model, X_val, prepared.y_val, prepared.target_scores_val, prepared.meta_val)

        selection = _selection_value(task_name, spec.prediction_mode, val_metrics)
        LOGGER.info("Validation results for %s: primary=%s, metrics=%s", candidate_name, selection, {key: val_metrics.get(key) for key in ('accuracy', 'balanced_accuracy', 'f1', 'roc_auc', 'macro_f1', 'mae', 'spearman')})
        if best_payload is None or selection > best_payload[0]:
            best_payload = (selection, model, spec, scaler, threshold, val_metrics, val_predictions)
            LOGGER.info("Current best model: %s", candidate_name)

    if best_payload is None:
        raise ValueError("No models were trained.")

    _, best_model, best_spec, best_scaler, threshold, val_metrics, val_predictions = best_payload
    LOGGER.info("Selected best %s model: %s", task_name, best_spec.name)
    prepared.preprocessor.feature_scaler = best_scaler
    prepared.preprocessor.model_prediction_mode = best_spec.prediction_mode
    prepared.preprocessor.decision_threshold = threshold
    X_test = best_scaler.transform(prepared.X_test) if best_scaler else prepared.X_test
    LOGGER.info("Evaluating best model on held-out test split")
    if task_name == "concentration":
        test_metrics, test_predictions = _evaluate_concentration(best_model, X_test, prepared.y_test, prepared.meta_test, threshold or 0.5)
    elif best_spec.prediction_mode == "regressor":
        test_metrics, test_predictions = _evaluate_stress_regressor(best_model, X_test, prepared.y_test, prepared.target_scores_test, prepared.meta_test)
    else:
        test_metrics, test_predictions = _evaluate_stress_classifier(best_model, X_test, prepared.y_test, prepared.target_scores_test, prepared.meta_test)
    val_metrics["quality_control"] = prepared.quality_reports["val"]
    test_metrics["quality_control"] = prepared.quality_reports["test"]
    LOGGER.info("Test results for %s: %s", best_spec.name, {key: test_metrics.get(key) for key in ('accuracy', 'balanced_accuracy', 'f1', 'roc_auc', 'macro_f1', 'mae', 'spearman')})
    return TrainingResult(best_model, best_spec.name, best_spec.prediction_mode, prepared.preprocessor, threshold, val_metrics, val_predictions, test_metrics, test_predictions)


def save_training_artifacts(task_name: str, result: TrainingResult, artifacts_root: Path) -> None:
    task_root = ensure_dir(artifacts_root / task_name)
    plots_root = ensure_dir(task_root / "plots")
    LOGGER.info("Saving %s artifacts to %s", task_name, task_root)
    previous_summary = load_json(task_root / "summary.json") if (task_root / "summary.json").exists() else None

    save_pickle({"model": result.model, "model_name": result.model_name, "prediction_mode": result.prediction_mode, "decision_threshold": result.decision_threshold}, task_root / "model.pkl")
    save_pickle(result.preprocessor, task_root / "preprocessor.pkl")
    save_json(result.val_metrics, task_root / "metrics_val.json")
    save_json(result.test_metrics, task_root / "metrics_test.json")
    save_dataframe(result.val_predictions, task_root / "predictions_val.csv")
    save_dataframe(result.test_predictions, task_root / "predictions_test.csv")

    recording_val_predictions, recording_val_metrics, pairwise_val_predictions, pairwise_val_metrics = _recording_and_pairwise_outputs(task_name, result.val_predictions)
    recording_test_predictions, recording_test_metrics, pairwise_test_predictions, pairwise_test_metrics = _recording_and_pairwise_outputs(task_name, result.test_predictions)
    save_dataframe(recording_val_predictions, task_root / "predictions_recording_val.csv")
    save_dataframe(recording_test_predictions, task_root / "predictions_recording_test.csv")
    save_dataframe(pairwise_val_predictions, task_root / "predictions_pairwise_val.csv")
    save_dataframe(pairwise_test_predictions, task_root / "predictions_pairwise_test.csv")
    save_json(recording_val_metrics, task_root / "metrics_recording_val.json")
    save_json(recording_test_metrics, task_root / "metrics_recording_test.json")
    save_json(pairwise_val_metrics, task_root / "metrics_pairwise_val.json")
    save_json(pairwise_test_metrics, task_root / "metrics_pairwise_test.json")

    summary = summary_payload(task_name, result.model_name, result.val_metrics, result.test_metrics)
    summary["validation_recording"] = recording_val_metrics
    summary["test_recording"] = recording_test_metrics
    summary["validation_pairwise"] = pairwise_val_metrics
    summary["test_pairwise"] = pairwise_test_metrics
    save_json(summary, task_root / "summary.json")
    if previous_summary is not None:
        save_json(comparison_payload(previous_summary, summary), task_root / "comparison.json")

    if task_name == "concentration":
        save_confusion_matrix_plot(result.test_metrics["confusion_matrix"], ["rest", "concentration"], plots_root / "confusion_matrix.png", "Concentration Confusion Matrix")
        save_confusion_matrix_plot(result.test_metrics["confusion_matrix"], ["rest", "concentration"], task_root / "confusion_matrix.png", "Concentration Confusion Matrix")
        save_distribution_plot(result.test_predictions["concentration_probability"].to_numpy(), plots_root / "probability_distribution.png", "Concentration Probability Distribution", "P(concentration)")
        save_distribution_plot(result.test_predictions["concentration_probability"].to_numpy(), task_root / "probability_distribution.png", "Concentration Probability Distribution", "P(concentration)")
        if result.test_metrics.get("roc_auc") is not None:
            save_roc_curve_plot(result.test_predictions["true_label"].to_numpy(), result.test_predictions["concentration_probability"].to_numpy(), plots_root / "roc_curve.png")
            save_roc_curve_plot(result.test_predictions["true_label"].to_numpy(), result.test_predictions["concentration_probability"].to_numpy(), task_root / "roc_curve.png")
    else:
        save_confusion_matrix_plot(result.test_metrics["confusion_matrix"], list(STRESS_CLASS_NAMES), plots_root / "confusion_matrix.png", "Stress Confusion Matrix")
        save_confusion_matrix_plot(result.test_metrics["confusion_matrix"], list(STRESS_CLASS_NAMES), task_root / "confusion_matrix.png", "Stress Confusion Matrix")
        save_distribution_plot(result.test_predictions["stress_score"].to_numpy(), plots_root / "score_distribution.png", "Stress Score Distribution", "Stress score")
        save_distribution_plot(result.test_predictions["stress_score"].to_numpy(), task_root / "score_distribution.png", "Stress Score Distribution", "Stress score")
        save_score_scatter_plot(result.test_predictions["true_score"].to_numpy(), result.test_predictions["predicted_score"].to_numpy(), plots_root / "score_scatter.png")
        save_score_scatter_plot(result.test_predictions["true_score"].to_numpy(), result.test_predictions["predicted_score"].to_numpy(), task_root / "score_scatter.png")
    LOGGER.info("Saved model, preprocessor, metrics, predictions, and plots for %s", task_name)


def terminal_summary(task_name: str, result: TrainingResult, split: SubjectSplit) -> str:
    metric_name = "roc_auc" if task_name == "concentration" else "spearman"
    _, recording_test_metrics, _, pairwise_test_metrics = _recording_and_pairwise_outputs(task_name, result.test_predictions)
    recording_metric_name = "roc_auc" if task_name == "concentration" else "spearman"
    return (
        f"{task_name} complete\n"
        f"model: {result.model_name}\n"
        f"train subjects: {len(split.train_subjects)} | val subjects: {len(split.val_subjects)} | test subjects: {len(split.test_subjects)}\n"
        f"validation {metric_name}: {result.val_metrics.get(metric_name)}\n"
        f"test {metric_name}: {result.test_metrics.get(metric_name)}\n"
        f"recording-level test {recording_metric_name}: {recording_test_metrics.get(recording_metric_name)}\n"
        f"pairwise test accuracy: {pairwise_test_metrics.get('pair_accuracy')}\n"
        f"test metrics: {result.test_metrics}"
    )
