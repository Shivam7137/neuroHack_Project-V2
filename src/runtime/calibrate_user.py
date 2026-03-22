"""Headless user calibration entrypoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.runtime.calibration_controller import CalibrationController, CalibrationError, build_calibration_source
from src.runtime.constants import RUNTIME_SETUP_CHANNELS


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the adaptive EEG setup flow and save a user profile.")
    parser.add_argument("--source", choices=("generator", "cyton"), required=True, help="Setup source.")
    parser.add_argument("--serial-port", default="COM5", help="Cyton serial port, for example COM5.")
    parser.add_argument("--user-id", required=True, help="Stable user identifier for saved setup artifacts.")
    parser.add_argument("--artifacts-root", default="artifacts/runtime_v1", help="Runtime artifacts bundle root.")
    parser.add_argument("--chunk-seconds", type=float, default=0.5, help="Generator chunk duration in seconds.")
    parser.add_argument("--seed", type=int, default=42, help="Generator seed.")
    args = parser.parse_args()

    artifacts_root = Path(args.artifacts_root).expanduser().resolve()
    source = build_calibration_source(
        source_type=args.source,
        serial_port=args.serial_port,
        chunk_seconds=args.chunk_seconds,
        seed=args.seed,
        channel_names=list(RUNTIME_SETUP_CHANNELS),
    )
    controller = CalibrationController(
        source=source,
        source_type=args.source,
        user_id=args.user_id,
        artifacts_root=artifacts_root,
        expected_channels=list(RUNTIME_SETUP_CHANNELS),
    )
    try:
        result = controller.run()
    except CalibrationError as exc:
        raise SystemExit(str(exc)) from exc

    payload = {
        "session_id": result.session_id,
        "profile_id": result.user_profile.profile_id,
        "user_id": result.user_profile.user_id,
        "profile_path": str(result.profile_files["profile_json"]),
        "profile_pickle_path": str(result.profile_files["profile_pickle"]),
        "session_summary_path": str(result.session_summary_path),
        "source_type": result.user_profile.source_type,
        "channel_names": result.user_profile.channel_names,
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
