"""Train a shared-channel runtime bundle for setup and live Cyton use."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.config import Settings
from src.datasets.eegmat_loader import EEGMATLoader
from src.datasets.stress_local_loader import StressLocalLoader
from src.runtime.constants import RUNTIME_SETUP_CHANNELS
from src.runtime.run_engine import validate_runtime_compatibility
from src.training.common import prepare_data, save_training_artifacts, train_select_and_evaluate
from src.utils.io import save_json
from src.utils.logging_utils import get_logger
from src.utils.seed import set_global_seed


def _parse_channels(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a runtime-compatible concentration/stress bundle.")
    parser.add_argument(
        "--channel-names",
        default=",".join(RUNTIME_SETUP_CHANNELS),
        help="Comma-separated canonical channel list used for both concentration and stress.",
    )
    parser.add_argument(
        "--artifacts-root",
        default="artifacts/runtime_v1",
        help="Output directory for the runtime bundle.",
    )
    parser.add_argument(
        "--concentration-cleanup",
        default="light",
        help="Cleanup profile used for concentration runtime training.",
    )
    parser.add_argument(
        "--stress-cleanup",
        default="heavy",
        help="Cleanup profile used for stress runtime training.",
    )
    args = parser.parse_args()

    logger = get_logger("train_runtime_bundle")
    settings = Settings()
    settings.artifacts_root = Path(args.artifacts_root).expanduser().resolve()
    settings.ensure_roots()
    set_global_seed(settings.random_seed)

    channel_names = _parse_channels(args.channel_names)
    logger.info("Training runtime bundle at %s", settings.artifacts_root)
    logger.info("Runtime channels: %s", ", ".join(channel_names))

    settings.cleanup_level = args.concentration_cleanup
    concentration_prepared, concentration_split = prepare_data(
        EEGMATLoader(settings=settings),
        task_name="concentration",
        settings=settings,
        channel_names_override=channel_names,
    )
    concentration_result = train_select_and_evaluate(
        concentration_prepared,
        candidate_names=settings.concentration_candidates,
        task_name="concentration",
    )
    save_training_artifacts("concentration", concentration_result, settings.artifacts_root)
    logger.info(
        "Saved runtime concentration bundle with %s channels for %s train subjects",
        len(channel_names),
        len(concentration_split.train_subjects),
    )

    settings.cleanup_level = args.stress_cleanup
    stress_prepared, stress_split = prepare_data(
        StressLocalLoader(settings=settings),
        task_name="stress",
        settings=settings,
        channel_names_override=channel_names,
    )
    stress_result = train_select_and_evaluate(
        stress_prepared,
        candidate_names=settings.stress_candidates,
        task_name="stress",
    )
    save_training_artifacts("stress", stress_result, settings.artifacts_root)
    logger.info(
        "Saved runtime stress bundle with %s channels for %s train subjects",
        len(channel_names),
        len(stress_split.train_subjects),
    )

    from src.runtime.baseline import BaselineInference

    validate_runtime_compatibility(
        BaselineInference(artifacts_root=settings.artifacts_root),
        runtime_channel_names=channel_names,
    )
    save_json(
        {
            "bundle_name": "runtime_v1",
            "channel_names": channel_names,
            "artifacts_root": str(settings.artifacts_root),
            "concentration_cleanup": args.concentration_cleanup,
            "stress_cleanup": args.stress_cleanup,
            "tasks": ["concentration", "stress"],
        },
        settings.artifacts_root / "runtime_bundle_summary.json",
    )
    logger.info("Runtime bundle validation passed for %s", channel_names)


if __name__ == "__main__":
    main()
