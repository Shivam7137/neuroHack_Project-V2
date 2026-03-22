"""Aggregate saved training artifacts into one summary report."""

from __future__ import annotations

import json
from pathlib import Path

from src.config import get_settings
from src.utils.io import load_json, save_json
from src.utils.logging_utils import get_logger


def _task_summary(task_root: Path) -> dict:
    required = [
        task_root / "model.pkl",
        task_root / "preprocessor.pkl",
        task_root / "metrics_val.json",
        task_root / "metrics_test.json",
        task_root / "predictions_test.csv",
        task_root / "plots" / "confusion_matrix.png",
    ]
    return {
        "exists": task_root.exists(),
        "complete": all(path.exists() for path in required),
        "missing_files": [str(path) for path in required if not path.exists()],
        "validation": load_json(task_root / "metrics_val.json") if (task_root / "metrics_val.json").exists() else {},
        "test": load_json(task_root / "metrics_test.json") if (task_root / "metrics_test.json").exists() else {},
    }


def _artifact_summary(task_root: Path) -> dict:
    required = [
        task_root / "model.pkl",
        task_root / "preprocessor.pkl",
        task_root / "metrics_val.json",
        task_root / "metrics_test.json",
        task_root / "predictions_test.csv",
        task_root / "plots" / "confusion_matrix.png",
    ]
    return {
        "exists": task_root.exists(),
        "complete": all(path.exists() for path in required),
        "missing_files": [str(path) for path in required if not path.exists()],
        "validation": load_json(task_root / "metrics_val.json") if (task_root / "metrics_val.json").exists() else {},
        "test": load_json(task_root / "metrics_test.json") if (task_root / "metrics_test.json").exists() else {},
    }


def main() -> None:
    settings = get_settings()
    logger = get_logger("evaluate_all")
    summary = {
        "concentration": _task_summary(settings.artifacts_root / "concentration"),
        "stress": _task_summary(settings.artifacts_root / "stress"),
        "artifact": _artifact_summary(settings.artifacts_root / "artifact"),
    }
    save_json(summary, settings.artifacts_root / "summary_all.json")
    logger.info(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
