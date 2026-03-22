"""CLI to replay a saved recording through the runtime engine."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.runtime.engine import StreamingEngine
from src.runtime.sources.playback_source import PlaybackSource


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay a saved EEG recording through the runtime engine.")
    parser.add_argument("--input", required=True, help="Path to a playback .npz, .npy, or .csv file.")
    parser.add_argument("--frames", type=int, default=6, help="Maximum number of chunks to replay before stopping.")
    parser.add_argument("--raw-only", action="store_true", help="Print raw frame metadata without scoring.")
    args = parser.parse_args()

    source = PlaybackSource(Path(args.input))
    engine = None if args.raw_only else StreamingEngine()
    source.start()
    try:
        for _ in range(args.frames):
            frame = source.read_frame()
            if args.raw_only:
                print(
                    json.dumps(
                        {
                            "timestamp": frame.timestamp,
                            "source": frame.source,
                            "shape": list(frame.data.shape),
                            "sample_rate": frame.sample_rate,
                            "channel_names": frame.channel_names,
                        }
                    )
                )
                continue
            outputs = engine.process_frame(frame)
            for output in outputs:
                print(
                    json.dumps(
                        {
                            "timestamp": output.timestamp,
                            "source": output.source,
                            "concentration_score": output.concentration_score,
                            "concentration_probability": output.concentration_probability,
                            "stress_score": output.stress_score,
                            "stress_predicted_class": output.stress_predicted_class,
                            "quality_score": output.quality_score,
                            "quality_label": output.quality_label,
                        }
                    )
                )
    finally:
        source.stop()


if __name__ == "__main__":
    main()
