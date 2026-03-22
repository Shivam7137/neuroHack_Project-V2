"""Export a canonical playback recording to OpenBCI GUI and BrainFlow file formats."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.runtime.playback_tools import export_playback_formats


def main() -> None:
    parser = argparse.ArgumentParser(description="Export playback data to OpenBCI GUI and BrainFlow file formats.")
    parser.add_argument("--input", required=True, help="Path to a canonical .npz playback recording.")
    parser.add_argument(
        "--output-dir",
        default="artifacts/playback/exports",
        help="Directory where exported files will be written.",
    )
    parser.add_argument(
        "--prefix",
        default="sample_cyton_playback",
        help="Base filename prefix for exported files.",
    )
    args = parser.parse_args()

    result = export_playback_formats(Path(args.input), Path(args.output_dir), args.prefix)
    print(result.openbci_path)
    print(result.brainflow_path)


if __name__ == "__main__":
    main()
