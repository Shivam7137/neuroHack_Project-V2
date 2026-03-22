"""Metrics for artifact-quality classification."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from src.datasets.artifact_common import ARTIFACT_CLASS_NAMES, artifact_binary_target
from src.evaluation.metrics import _safe_float


def artifact_metrics(y_true: np.ndarray, y_pred: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    """Compute multiclass artifact metrics plus clean-vs-artifact rollups."""
    labels = list(range(len(ARTIFACT_CLASS_NAMES)))
    clean_index = 0
    noisy_probability = 1.0 - probabilities[:, clean_index]
    y_true_binary = np.asarray(artifact_binary_target(y_true), dtype=int)
    y_pred_binary = np.asarray(artifact_binary_target(y_pred), dtype=int)
    metrics: dict[str, Any] = {
        "accuracy": _safe_float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": _safe_float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": _safe_float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "mean_clean_probability": _safe_float(np.mean(probabilities[:, clean_index])),
        "mean_noisy_probability": _safe_float(np.mean(noisy_probability)),
        "binary_accuracy": _safe_float(accuracy_score(y_true_binary, y_pred_binary)),
        "binary_precision": _safe_float(precision_score(y_true_binary, y_pred_binary, zero_division=0)),
        "binary_recall": _safe_float(recall_score(y_true_binary, y_pred_binary, zero_division=0)),
        "binary_f1": _safe_float(f1_score(y_true_binary, y_pred_binary, zero_division=0)),
        "binary_balanced_accuracy": _safe_float(balanced_accuracy_score(y_true_binary, y_pred_binary)),
        "binary_confusion_matrix": confusion_matrix(y_true_binary, y_pred_binary, labels=[0, 1]).tolist(),
    }
    return metrics


def artifact_per_subject(predictions: pd.DataFrame) -> list[dict[str, Any]]:
    """Compute per-subject artifact metrics."""
    results: list[dict[str, Any]] = []
    probability_columns = [f"probability_{name}" for name in ARTIFACT_CLASS_NAMES]
    for subject_id, frame in predictions.groupby("subject_id"):
        metrics = artifact_metrics(
            frame["true_label"].to_numpy(),
            frame["predicted_label"].to_numpy(),
            frame[probability_columns].to_numpy(),
        )
        metrics["subject_id"] = subject_id
        metrics["mean_quality_score"] = _safe_float(frame["quality_score"].mean())
        results.append(metrics)
    return results
