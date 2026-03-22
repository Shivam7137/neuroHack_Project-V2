"""CLI entrypoint for stress model training."""

from __future__ import annotations

import argparse

from src.config import get_settings
from src.datasets.stress_local_loader import StressLocalLoader
from src.preprocessing.cleanup import CLEANUP_LEVELS
from src.training.common import prepare_data, save_training_artifacts, terminal_summary, train_select_and_evaluate
from src.utils.logging_utils import get_logger
from src.utils.seed import set_global_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the stress decoder.")
    parser.add_argument(
        "--mode",
        choices=["classifier", "regressor"],
        default="regressor",
        help="Train the default regressor or the optional classifier benchmark.",
    )
    parser.add_argument(
        "--cleanup-level",
        choices=CLEANUP_LEVELS,
        default=None,
        help="Optional cleanup treatment to apply before feature extraction.",
    )
    args = parser.parse_args()

    logger = get_logger("train_stress")
    settings = get_settings()
    settings.cleanup_level = args.cleanup_level or settings.stress_cleanup_level
    set_global_seed(settings.random_seed)

    candidates = (
        settings.stress_candidates
        if args.mode == "regressor"
        else settings.stress_classifier_candidates
    )
    logger.info("Starting stress training")
    logger.info("STRESS_DATA_ROOT=%s", settings.stress_data_root)
    logger.info("ARTIFACTS_ROOT=%s", settings.artifacts_root)
    logger.info("Cleanup level=%s", settings.cleanup_level)
    logger.info("Mode=%s | Candidates=%s", args.mode, ", ".join(candidates))

    loader = StressLocalLoader(settings=settings)
    prepared, split = prepare_data(loader, task_name="stress", settings=settings)
    result = train_select_and_evaluate(prepared, candidate_names=tuple(candidates), task_name="stress")
    save_training_artifacts("stress", result, settings.artifacts_root)
    logger.info("\n%s", terminal_summary("stress", result, split))


if __name__ == "__main__":
    main()
