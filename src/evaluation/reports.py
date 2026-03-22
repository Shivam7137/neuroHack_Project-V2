"""Prediction tables and report assembly."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _group_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in ("subject_id", "session_id", "file_path") if column in frame.columns]


def _nearest_class_label(score: float, class_to_score: dict[int, float]) -> int:
    return int(min(class_to_score, key=lambda class_id: abs(class_to_score[class_id] - score)))


def concentration_predictions_frame(
    metadata: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> pd.DataFrame:
    """Build a concentration predictions DataFrame."""
    frame = metadata.reset_index(drop=True).copy()
    frame["true_label"] = y_true
    frame["predicted_label"] = y_pred
    frame["concentration_probability"] = probabilities
    frame["concentration_score"] = probabilities * 100.0
    frame["decision_threshold"] = threshold
    frame["predicted_positive"] = y_pred
    frame["observed_positive"] = y_true
    frame["calibration_error"] = np.abs(frame["concentration_probability"] - frame["observed_positive"])
    return frame


def stress_predictions_frame(
    metadata: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray | None,
    class_names: list[str],
    class_to_score: dict[int, float],
    predicted_scores: np.ndarray,
    true_scores: np.ndarray,
) -> pd.DataFrame:
    """Build a stress predictions DataFrame."""
    frame = metadata.reset_index(drop=True).copy()
    frame["true_label"] = y_true
    frame["predicted_label"] = y_pred
    frame["true_class_name"] = [class_names[int(label)] for label in y_true]
    frame["predicted_class_name"] = [class_names[int(label)] for label in y_pred]
    frame["compatibility_true_score"] = [class_to_score[int(label)] for label in y_true]
    frame["compatibility_predicted_score"] = [class_to_score[int(label)] for label in y_pred]
    frame["true_score"] = true_scores
    frame["predicted_score"] = predicted_scores
    if probabilities is not None:
        for class_index, class_name in enumerate(class_names):
            frame[f"probability_{class_name}"] = probabilities[:, class_index]
    if "rating" in frame.columns:
        true_ratings = pd.to_numeric(frame["rating"], errors="coerce")
        predicted_ratings = np.clip(np.rint(frame["predicted_score"] * 8.0 + 1.0), 1, 9).astype(int)
        frame["true_rating_level"] = true_ratings.map(lambda rating: np.nan if pd.isna(rating) else ("low" if rating <= 3 else "moderate" if rating <= 6 else "high"))
        frame["predicted_rating_level"] = predicted_ratings.map(lambda rating: "low" if rating <= 3 else "moderate" if rating <= 6 else "high")
    frame["stress_score"] = frame["predicted_score"] * 100.0
    return frame


def concentration_recording_predictions_frame(window_predictions: pd.DataFrame) -> pd.DataFrame:
    """Aggregate window predictions into one row per recording."""
    group_cols = _group_columns(window_predictions)
    frame = (
        window_predictions.groupby(group_cols, dropna=False)
        .agg(
            true_label=("true_label", "first"),
            raw_label=("raw_label", "first"),
            concentration_probability=("concentration_probability", "mean"),
            decision_threshold=("decision_threshold", "first"),
            n_windows=("true_label", "size"),
            sampling_rate=("sampling_rate", "first"),
            start_sample=("start_sample", "min"),
            end_sample=("end_sample", "max"),
        )
        .reset_index()
    )
    frame["predicted_label"] = (frame["concentration_probability"] >= frame["decision_threshold"]).astype(int)
    frame["concentration_score"] = frame["concentration_probability"] * 100.0
    frame["predicted_positive"] = frame["predicted_label"]
    frame["observed_positive"] = frame["true_label"]
    frame["calibration_error"] = np.abs(frame["concentration_probability"] - frame["true_label"])
    return frame


def stress_recording_predictions_frame(
    window_predictions: pd.DataFrame,
    class_names: list[str],
    class_to_score: dict[int, float],
) -> pd.DataFrame:
    """Aggregate window predictions into one row per recording."""
    group_cols = _group_columns(window_predictions)
    aggregations: dict[str, tuple[str, str]] = {
        "true_label": ("true_label", "first"),
        "raw_label": ("raw_label", "first"),
        "mapped_label": ("mapped_label", "first"),
        "true_class_name": ("true_class_name", "first"),
        "compatibility_true_score": ("compatibility_true_score", "first"),
        "true_score": ("true_score", "first"),
        "predicted_score": ("predicted_score", "mean"),
        "n_windows": ("true_label", "size"),
        "sampling_rate": ("sampling_rate", "first"),
        "start_sample": ("start_sample", "min"),
        "end_sample": ("end_sample", "max"),
    }
    for optional in ("source_dataset", "source_variant", "condition", "rating", "rating_normalized", "target_score", "true_rating_level"):
        if optional in window_predictions.columns:
            aggregations[optional] = (optional, "first")
    frame = window_predictions.groupby(group_cols, dropna=False).agg(**aggregations).reset_index()
    frame["predicted_label"] = frame["predicted_score"].map(lambda value: _nearest_class_label(float(value), class_to_score))
    frame["predicted_class_name"] = [class_names[int(label)] for label in frame["predicted_label"]]
    frame["compatibility_predicted_score"] = [class_to_score[int(label)] for label in frame["predicted_label"]]
    if "rating" in frame.columns:
        predicted_ratings = np.clip(np.rint(frame["predicted_score"] * 8.0 + 1.0), 1, 9).astype(int)
        frame["predicted_rating_level"] = pd.Series(predicted_ratings).map(
            lambda rating: "low" if rating <= 3 else "moderate" if rating <= 6 else "high"
        )
    frame["stress_score"] = frame["predicted_score"] * 100.0
    return frame


def concentration_pairwise_frame(recording_predictions: pd.DataFrame) -> pd.DataFrame:
    """Create within-subject baseline-vs-task ranking rows."""
    rows: list[dict[str, Any]] = []
    for subject_id, frame in recording_predictions.groupby("subject_id"):
        negatives = frame[frame["true_label"] == 0]
        positives = frame[frame["true_label"] == 1]
        for _, negative in negatives.iterrows():
            for _, positive in positives.iterrows():
                margin = float(positive["concentration_probability"] - negative["concentration_probability"])
                rows.append(
                    {
                        "subject_id": subject_id,
                        "lower_session_id": negative.get("session_id"),
                        "higher_session_id": positive.get("session_id"),
                        "lower_file_path": negative.get("file_path"),
                        "higher_file_path": positive.get("file_path"),
                        "lower_true_value": 0,
                        "higher_true_value": 1,
                        "lower_predicted_value": float(negative["concentration_probability"]),
                        "higher_predicted_value": float(positive["concentration_probability"]),
                        "predicted_margin": margin,
                        "pair_correct": int(margin > 0.0),
                    }
                )
    return pd.DataFrame(rows)


def stress_pairwise_frame(recording_predictions: pd.DataFrame) -> pd.DataFrame:
    """Create within-subject pairwise ranking rows for recordings with distinct true scores."""
    rows: list[dict[str, Any]] = []
    for subject_id, frame in recording_predictions.groupby("subject_id"):
        ordered = frame.sort_values(["true_score", "session_id"], kind="stable").reset_index(drop=True)
        for lower_index in range(len(ordered)):
            lower = ordered.iloc[lower_index]
            for higher_index in range(lower_index + 1, len(ordered)):
                higher = ordered.iloc[higher_index]
                if float(higher["true_score"]) <= float(lower["true_score"]):
                    continue
                margin = float(higher["predicted_score"] - lower["predicted_score"])
                rows.append(
                    {
                        "subject_id": subject_id,
                        "lower_session_id": lower.get("session_id"),
                        "higher_session_id": higher.get("session_id"),
                        "lower_file_path": lower.get("file_path"),
                        "higher_file_path": higher.get("file_path"),
                        "lower_true_value": float(lower["true_score"]),
                        "higher_true_value": float(higher["true_score"]),
                        "lower_predicted_value": float(lower["predicted_score"]),
                        "higher_predicted_value": float(higher["predicted_score"]),
                        "predicted_margin": margin,
                        "pair_correct": int(margin > 0.0),
                    }
                )
    return pd.DataFrame(rows)


def summary_payload(task_name: str, model_name: str, val_metrics: dict[str, Any], test_metrics: dict[str, Any]) -> dict[str, Any]:
    """Create a top-level summary payload."""
    return {
        "task": task_name,
        "model_name": model_name,
        "validation": val_metrics,
        "test": test_metrics,
    }


def comparison_payload(previous_summary: dict[str, Any], current_summary: dict[str, Any]) -> dict[str, Any]:
    """Compare the latest summary against an existing baseline artifact."""
    fields = ("roc_auc", "balanced_accuracy", "f1", "spearman", "mae", "macro_f1")
    payload: dict[str, Any] = {"baseline_model": previous_summary.get("model_name"), "current_model": current_summary.get("model_name"), "delta": {}}
    for split_name in ("validation", "test"):
        split_delta: dict[str, Any] = {}
        previous = previous_summary.get(split_name, {})
        current = current_summary.get(split_name, {})
        for field in fields:
            if field in previous and field in current and previous.get(field) is not None and current.get(field) is not None:
                split_delta[field] = float(current[field] - previous[field])
        payload["delta"][split_name] = split_delta
    return payload
