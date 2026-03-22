"""CLI for creating a calibration profile from one calibration recording."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.preprocessing.calibration import build_calibration_profile, save_calibration_profile


def _load_signal(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        return np.asarray(np.load(path), dtype=float)
    if path.suffix.lower() == ".npz":
        payload = np.load(path, allow_pickle=True)
        key = "signal" if "signal" in payload else list(payload.keys())[0]
        return np.asarray(payload[key], dtype=float)
    if path.suffix.lower() == ".csv":
        return np.loadtxt(path, delimiter=",", dtype=float)
    raise ValueError(f"Unsupported calibration file format: {path.suffix}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a per-subject calibration profile.")
    parser.add_argument("--input", required=True, help="Path to a .npy, .npz, or .csv calibration recording.")
    parser.add_argument("--output", required=True, help="Path to the calibration profile .pkl file.")
    parser.add_argument("--sampling-rate", required=True, type=float, help="Sampling rate for the calibration recording.")
    parser.add_argument("--channel-names", required=True, help="Comma-separated channel names in recording order.")
    parser.add_argument("--subject-id", default="", help="Optional subject identifier.")
    parser.add_argument("--session-id", default="", help="Optional session identifier.")
    args = parser.parse_args()

    signal = _load_signal(Path(args.input))
    channel_names = [part.strip() for part in args.channel_names.split(",") if part.strip()]
    profile = build_calibration_profile(
        signal,
        sampling_rate=args.sampling_rate,
        channel_names=channel_names,
        subject_id=args.subject_id or None,
        session_id=args.session_id or None,
    )
    save_calibration_profile(profile, Path(args.output))


if __name__ == "__main__":
    main()
