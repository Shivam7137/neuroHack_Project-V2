"""Prediction tables and summaries for artifact-quality runs."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.datasets.artifact_common import ARTIFACT_CLASS_NAMES, artifact_binary_target
from src.evaluation.reports import _group_columns


def artifact_predictions_frame(
    metadata: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
) -> pd.DataFrame:
    """Build a window-level artifact predictions DataFrame."""
    frame = metadata.reset_index(drop=True).copy()
    frame["true_label"] = y_true
    frame["predicted_label"] = y_pred
    frame["true_class_name"] = [ARTIFACT_CLASS_NAMES[int(label)] for label in y_true]
    frame["predicted_class_name"] = [ARTIFACT_CLASS_NAMES[int(label)] for label in y_pred]
    for class_index, class_name in enumerate(ARTIFACT_CLASS_NAMES):
        frame[f"probability_{class_name}"] = probabilities[:, class_index]
    frame["binary_true_label"] = artifact_binary_target(y_true)
    frame["binary_predicted_label"] = artifact_binary_target(y_pred)
    frame["quality_score"] = probabilities[:, 0] * 100.0
    frame["quality_label"] = np.where(frame["predicted_label"] == 0, "clean", "noisy")
    return frame


def artifact_recording_predictions_frame(window_predictions: pd.DataFrame) -> pd.DataFrame:
    """Aggregate window predictions into one row per recording."""
    group_cols = _group_columns(window_predictions)
    aggregations: dict[str, tuple[str, str]] = {
        "true_label": ("true_label", "first"),
        "raw_label": ("raw_label", "first"),
        "mapped_label": ("mapped_label", "first"),
        "true_class_name": ("true_class_name", "first"),
        "quality_score": ("quality_score", "mean"),
        "sampling_rate": ("sampling_rate", "first"),
        "start_sample": ("start_sample", "min"),
        "end_sample": ("end_sample", "max"),
        "n_windows": ("true_label", "size"),
    }
    for class_name in ARTIFACT_CLASS_NAMES:
        aggregations[f"probability_{class_name}"] = (f"probability_{class_name}", "mean")
    for optional in (
        "source_dataset",
        "augmentation_source",
        "noise_scale",
        "pyprep_bad_channels",
        "quality_reasons",
        "pyprep_status",
        "autoreject_status",
    ):
        if optional in window_predictions.columns:
            aggregations[optional] = (optional, "first")
    frame = window_predictions.groupby(group_cols, dropna=False).agg(**aggregations).reset_index()
    probability_columns = [f"probability_{name}" for name in ARTIFACT_CLASS_NAMES]
    frame["predicted_label"] = np.argmax(frame[probability_columns].to_numpy(), axis=1)
    frame["predicted_class_name"] = [ARTIFACT_CLASS_NAMES[int(label)] for label in frame["predicted_label"]]
    frame["binary_true_label"] = artifact_binary_target(frame["true_label"].to_numpy())
    frame["binary_predicted_label"] = artifact_binary_target(frame["predicted_label"].to_numpy())
    frame["quality_label"] = np.where(frame["predicted_label"] == 0, "clean", "noisy")
    return frame


def artifact_summary_payload(model_name: str, val_metrics: dict[str, Any], test_metrics: dict[str, Any]) -> dict[str, Any]:
    """Create the top-level artifact summary payload."""
    return {
        "task": "artifact",
        "model_name": model_name,
        "validation": val_metrics,
        "test": test_metrics,
    }
