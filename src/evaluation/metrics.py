"""Metric computation for concentration and stress tasks."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    result = float(value)
    return None if np.isnan(result) else result


def concentration_metrics(y_true: np.ndarray, y_pred: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "accuracy": _safe_float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": _safe_float(balanced_accuracy_score(y_true, y_pred)),
        "precision": _safe_float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": _safe_float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": _safe_float(f1_score(y_true, y_pred, zero_division=0)),
        "brier_score": _safe_float(brier_score_loss(y_true, probabilities)),
        "mean_probability": _safe_float(np.mean(probabilities)),
        "positive_rate": _safe_float(np.mean(y_true)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }
    metrics["roc_auc"] = _safe_float(roc_auc_score(y_true, probabilities)) if len(np.unique(y_true)) > 1 else None
    return metrics


def stress_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    true_scores: np.ndarray,
    predicted_scores: np.ndarray,
    ratings: np.ndarray | None = None,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "accuracy": _safe_float(accuracy_score(y_true, y_pred)),
        "macro_f1": _safe_float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "mae": _safe_float(np.mean(np.abs(true_scores - predicted_scores))),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3]).tolist(),
        "quadratic_weighted_kappa": _safe_float(cohen_kappa_score(y_true, y_pred, weights="quadratic")),
    }
    metrics["spearman"] = _safe_float(spearmanr(true_scores, predicted_scores).statistic if len(true_scores) > 1 else np.nan)
    if ratings is not None:
        numeric_ratings = pd.to_numeric(pd.Series(ratings), errors="coerce").to_numpy(dtype=float)
        valid_mask = ~np.isnan(numeric_ratings)
        if np.any(valid_mask):
            predicted_ratings = np.clip(np.rint(predicted_scores[valid_mask] * 8.0 + 1.0), 1, 9).astype(int)
            rating_true = np.asarray(["low" if rating <= 3 else "moderate" if rating <= 6 else "high" for rating in numeric_ratings[valid_mask]], dtype=object)
            rating_pred = np.asarray(["low" if rating <= 3 else "moderate" if rating <= 6 else "high" for rating in predicted_ratings], dtype=object)
            labels = ["low", "moderate", "high"]
            metrics["rating_level_accuracy"] = _safe_float(np.mean(rating_true == rating_pred))
            metrics["rating_level_confusion_matrix"] = confusion_matrix(rating_true, rating_pred, labels=labels).tolist()
    return metrics


def concentration_per_subject(predictions: pd.DataFrame) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for subject_id, frame in predictions.groupby("subject_id"):
        metrics = concentration_metrics(
            frame["true_label"].to_numpy(),
            frame["predicted_label"].to_numpy(),
            frame["concentration_probability"].to_numpy(),
        )
        metrics["subject_id"] = subject_id
        metrics["mean_calibration_error"] = _safe_float(frame["calibration_error"].mean())
        results.append(metrics)
    return results


def stress_per_subject(predictions: pd.DataFrame) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for subject_id, frame in predictions.groupby("subject_id"):
        ratings = frame["rating"].to_numpy() if "rating" in frame.columns else None
        metrics = stress_metrics(
            frame["true_label"].to_numpy(),
            frame["predicted_label"].to_numpy(),
            frame["true_score"].to_numpy(),
            frame["predicted_score"].to_numpy(),
            ratings=ratings,
        )
        metrics["subject_id"] = subject_id
        results.append(metrics)
    return results


def pairwise_ranking_metrics(pairwise_frame: pd.DataFrame) -> dict[str, Any]:
    """Summarize within-subject pairwise ordering accuracy."""
    if pairwise_frame.empty:
        return {
            "pair_count": 0,
            "subject_count": 0,
            "pair_accuracy": None,
            "mean_margin": None,
            "median_margin": None,
            "tie_rate": None,
            "per_subject": [],
        }

    per_subject: list[dict[str, Any]] = []
    for subject_id, frame in pairwise_frame.groupby("subject_id"):
        per_subject.append(
            {
                "subject_id": subject_id,
                "pair_count": int(len(frame)),
                "pair_accuracy": _safe_float(frame["pair_correct"].mean()),
                "mean_margin": _safe_float(frame["predicted_margin"].mean()),
                "median_margin": _safe_float(frame["predicted_margin"].median()),
                "tie_rate": _safe_float((frame["predicted_margin"] == 0.0).mean()),
            }
        )

    return {
        "pair_count": int(len(pairwise_frame)),
        "subject_count": int(pairwise_frame["subject_id"].nunique()),
        "pair_accuracy": _safe_float(pairwise_frame["pair_correct"].mean()),
        "mean_margin": _safe_float(pairwise_frame["predicted_margin"].mean()),
        "median_margin": _safe_float(pairwise_frame["predicted_margin"].median()),
        "tie_rate": _safe_float((pairwise_frame["predicted_margin"] == 0.0).mean()),
        "per_subject": per_subject,
    }
