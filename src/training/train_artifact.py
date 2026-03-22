"""CLI entrypoint for artifact-quality model training."""

from __future__ import annotations

import argparse

from src.config import get_settings
from src.training.artifact_pipeline import artifact_terminal_summary, prepare_artifact_data, save_artifact_artifacts, train_select_and_evaluate_artifact
from src.utils.logging_utils import get_logger
from src.utils.seed import set_global_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the artifact-quality sidecar model.")
    parser.parse_args()

    logger = get_logger("train_artifact")
    settings = get_settings()
    settings.cleanup_level = settings.artifact_cleanup_level
    set_global_seed(settings.random_seed)
    logger.info("Starting artifact-quality training")
    logger.info("TUAR_ROOT=%s", settings.tuar_root)
    logger.info("EEGDENOISENET_ROOT=%s", settings.eegdenoisenet_root)
    logger.info("ARTIFACTS_ROOT=%s", settings.artifacts_root)
    logger.info("Candidates=%s", ", ".join(settings.artifact_candidates))

    prepared, split = prepare_artifact_data(settings)
    result = train_select_and_evaluate_artifact(prepared, candidate_names=settings.artifact_candidates)
    save_artifact_artifacts(result, settings.artifacts_root)
    logger.info("\n%s", artifact_terminal_summary(result, split))


if __name__ == "__main__":
    main()
