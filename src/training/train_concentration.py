"""CLI entrypoint for concentration model training."""

from __future__ import annotations

import argparse

from src.config import get_settings
from src.datasets.eegmat_loader import EEGMATLoader
from src.preprocessing.cleanup import CLEANUP_LEVELS
from src.training.common import prepare_data, save_training_artifacts, terminal_summary, train_select_and_evaluate
from src.utils.logging_utils import get_logger
from src.utils.seed import set_global_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the concentration decoder.")
    parser.add_argument(
        "--cleanup-level",
        choices=CLEANUP_LEVELS,
        default=None,
        help="Optional cleanup treatment to apply before feature extraction.",
    )
    args = parser.parse_args()

    logger = get_logger("train_concentration")
    settings = get_settings()
    settings.cleanup_level = args.cleanup_level or settings.concentration_cleanup_level
    set_global_seed(settings.random_seed)
    logger.info("Starting concentration training")
    logger.info("EEGMAT_ROOT=%s", settings.eegmat_root)
    logger.info("ARTIFACTS_ROOT=%s", settings.artifacts_root)
    logger.info("Cleanup level=%s", settings.cleanup_level)
    logger.info("Candidates=%s", ", ".join(settings.concentration_candidates))

    loader = EEGMATLoader(settings=settings)
    prepared, split = prepare_data(loader, task_name="concentration", settings=settings)
    result = train_select_and_evaluate(
        prepared,
        candidate_names=settings.concentration_candidates,
        task_name="concentration",
    )
    save_training_artifacts("concentration", result, settings.artifacts_root)
    logger.info("\n%s", terminal_summary("concentration", result, split))


if __name__ == "__main__":
    main()
