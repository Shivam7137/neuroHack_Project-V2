"""Generate a sample canonical playback recording."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.runtime.playback_tools import generate_playback_npz


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a sample canonical playback recording.")
    parser.add_argument(
        "--output",
        default="artifacts/playback/sample_cyton_playback.npz",
        help="Path to the output .npz file.",
    )
    parser.add_argument("--duration-sec", type=float, default=12.0, help="Duration of the generated recording.")
    parser.add_argument("--concentration", type=float, default=0.7, help="Synthetic concentration level in [0, 1].")
    parser.add_argument("--stress", type=float, default=0.25, help="Synthetic stress level in [0, 1].")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible output.")
    args = parser.parse_args()

    output_path = generate_playback_npz(
        Path(args.output),
        duration_sec=float(args.duration_sec),
        concentration=float(args.concentration),
        stress=float(args.stress),
        seed=int(args.seed),
    )
    print(output_path)


if __name__ == "__main__":
    main()
