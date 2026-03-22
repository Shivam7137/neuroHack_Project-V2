"""Plotting helpers for saved reports."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.utils.io import ensure_dir


def save_confusion_matrix_plot(confusion: list[list[int]], labels: list[str], path: Path, title: str) -> None:
    """Save a confusion matrix heatmap."""
    ensure_dir(path.parent)
    matrix = np.asarray(confusion)
    fig, axis = plt.subplots(figsize=(5, 4))
    image = axis.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=axis)
    axis.set_xticks(range(len(labels)), labels=labels)
    axis.set_yticks(range(len(labels)), labels=labels)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    axis.set_title(title)
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            axis.text(col, row, str(matrix[row, col]), ha="center", va="center", color="black")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_roc_curve_plot(y_true: np.ndarray, probabilities: np.ndarray, path: Path) -> None:
    """Save a ROC curve plot."""
    from sklearn.metrics import RocCurveDisplay

    ensure_dir(path.parent)
    fig, axis = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_predictions(y_true, probabilities, ax=axis)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_distribution_plot(values: np.ndarray, path: Path, title: str, xlabel: str) -> None:
    """Save a simple histogram plot."""
    ensure_dir(path.parent)
    fig, axis = plt.subplots(figsize=(5, 4))
    axis.hist(values, bins=20, color="#2367a2", alpha=0.85)
    axis.set_title(title)
    axis.set_xlabel(xlabel)
    axis.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_score_scatter_plot(true_scores: np.ndarray, predicted_scores: np.ndarray, path: Path) -> None:
    """Save a true-vs-predicted scatter plot."""
    ensure_dir(path.parent)
    fig, axis = plt.subplots(figsize=(5, 4))
    axis.scatter(true_scores, predicted_scores, alpha=0.7)
    axis.plot([0, 1], [0, 1], linestyle="--", color="gray")
    axis.set_xlabel("True ordinal score")
    axis.set_ylabel("Predicted ordinal score")
    axis.set_title("Stress Score Scatter")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
