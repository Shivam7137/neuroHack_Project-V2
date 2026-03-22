"""Benchmark cleanup treatment levels for concentration and stress training."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from src.config import Settings
from src.datasets.eegmat_loader import EEGMATLoader
from src.datasets.stress_local_loader import StressLocalLoader
from src.preprocessing.cleanup import CLEANUP_LEVELS
from src.training.common import prepare_data, save_training_artifacts, train_select_and_evaluate
from src.utils.io import ensure_dir, save_json
from src.utils.logging_utils import get_logger
from src.utils.seed import set_global_seed

LOGGER = get_logger("training.benchmark_cleanup")


def _task_metrics(task_name: str, metrics: dict[str, Any]) -> dict[str, Any]:
    if task_name == "concentration":
        keys = ("roc_auc", "balanced_accuracy", "f1")
    else:
        keys = ("spearman", "mae", "macro_f1")
    return {key: metrics.get(key) for key in keys}


def _run_profile(profile: str, base_settings: Settings) -> dict[str, Any]:
    profile_root = base_settings.artifacts_root / "cleanup_benchmark" / profile
    settings = replace(base_settings, artifacts_root=profile_root, cleanup_level=profile)
    settings.ensure_roots()
    set_global_seed(settings.random_seed)

    concentration_loader = EEGMATLoader(settings=settings)
    concentration_prepared, concentration_split = prepare_data(concentration_loader, task_name="concentration", settings=settings)
    concentration_result = train_select_and_evaluate(concentration_prepared, settings.concentration_candidates, task_name="concentration")
    save_training_artifacts("concentration", concentration_result, settings.artifacts_root)

    stress_loader = StressLocalLoader(settings=settings)
    stress_prepared, stress_split = prepare_data(stress_loader, task_name="stress", settings=settings)
    stress_result = train_select_and_evaluate(stress_prepared, settings.stress_candidates, task_name="stress")
    save_training_artifacts("stress", stress_result, settings.artifacts_root)

    return {
        "cleanup_level": profile,
        "artifacts_root": str(profile_root),
        "concentration": {
            "subjects": {
                "train": len(concentration_split.train_subjects),
                "val": len(concentration_split.val_subjects),
                "test": len(concentration_split.test_subjects),
            },
            "validation": _task_metrics("concentration", concentration_result.val_metrics),
            "test": _task_metrics("concentration", concentration_result.test_metrics),
        },
        "stress": {
            "subjects": {
                "train": len(stress_split.train_subjects),
                "val": len(stress_split.val_subjects),
                "test": len(stress_split.test_subjects),
            },
            "validation": _task_metrics("stress", stress_result.val_metrics),
            "test": _task_metrics("stress", stress_result.test_metrics),
        },
    }


def _compute_delta(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    delta: dict[str, Any] = {}
    for task_name, metric_keys in {"concentration": ("roc_auc", "balanced_accuracy", "f1"), "stress": ("spearman", "mae", "macro_f1")}.items():
        task_delta: dict[str, Any] = {}
        for split_name in ("validation", "test"):
            split_delta: dict[str, Any] = {}
            for key in metric_keys:
                base_value = baseline[task_name][split_name].get(key)
                candidate_value = candidate[task_name][split_name].get(key)
                if base_value is not None and candidate_value is not None:
                    split_delta[key] = float(candidate_value - base_value)
            task_delta[split_name] = split_delta
        delta[task_name] = task_delta
    return delta


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark cleanup treatment levels for concentration and stress.")
    parser.add_argument(
        "--profiles",
        default="none,light,medium,heavy",
        help="Comma-separated cleanup profiles to benchmark.",
    )
    args = parser.parse_args()

    requested_profiles = [part.strip().lower() for part in args.profiles.split(",") if part.strip()]
    invalid = [profile for profile in requested_profiles if profile not in CLEANUP_LEVELS]
    if invalid:
        raise ValueError(f"Unsupported cleanup profiles: {invalid}. Expected choices from {CLEANUP_LEVELS}.")

    base_settings = Settings()
    base_settings.ensure_roots()
    benchmark_root = ensure_dir(base_settings.artifacts_root / "cleanup_benchmark")
    results: list[dict[str, Any]] = []
    for profile in requested_profiles:
        LOGGER.info("Running cleanup benchmark profile=%s", profile)
        results.append(_run_profile(profile, base_settings))

    baseline = next((result for result in results if result["cleanup_level"] == "none"), None)
    summary = {"profiles": results}
    if baseline is not None:
        summary["delta_vs_none"] = {
            result["cleanup_level"]: _compute_delta(baseline, result)
            for result in results
            if result["cleanup_level"] != "none"
        }
    summary_path = benchmark_root / "benchmark_summary.json"
    save_json(summary, summary_path)
    LOGGER.info(json.dumps(summary, indent=2))
    LOGGER.info("Cleanup benchmark summary written to %s", summary_path)


if __name__ == "__main__":
    main()
