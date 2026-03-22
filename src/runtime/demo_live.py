"""CLI for a short live Cyton runtime demo."""

from __future__ import annotations

import argparse
import json

from src.runtime.engine import StreamingEngine
from src.runtime.sources.cyton_source import CytonSource


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a short live Cyton demo through the runtime engine.")
    parser.add_argument("--serial-port", required=True, help="Cyton dongle serial port, for example COM5.")
    parser.add_argument("--frames", type=int, default=5, help="Number of runtime chunks to read before exiting.")
    parser.add_argument("--raw-only", action="store_true", help="Only print raw frame metadata without scoring.")
    args = parser.parse_args()

    source = CytonSource(serial_port=args.serial_port)
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
            if not outputs:
                continue
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
