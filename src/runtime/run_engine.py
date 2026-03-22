"""Unified streaming CLI for synthetic, playback, and live Cyton runtime sources."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterator

from src.runtime import AdaptationLayer, BaselineInference, DecisionEngine, DecisionPostprocessor, SourceRouter
from src.runtime.constants import CYTON_CHANNELS
from src.runtime.sources import CytonSource, EEGSource, PlaybackSource, SyntheticSource


def build_source(args: argparse.Namespace) -> EEGSource:
    """Build one runtime source from parsed CLI args."""
    if args.source == "synthetic":
        source = SyntheticSource(chunk_seconds=args.chunk_seconds)
        source.set_condition(args.concentration, args.stress)
        source.set_seed(args.seed)
        return source
    if args.source == "playback":
        if not args.input:
            raise ValueError("--input is required when --source playback.")
        return PlaybackSource(Path(args.input))
    if args.source == "cyton":
        if not args.serial_port:
            raise ValueError("--serial-port is required when --source cyton.")
        return CytonSource(serial_port=args.serial_port)
    raise ValueError(f"Unsupported source: {args.source}")


def build_router(source_name: str, source: EEGSource) -> SourceRouter:
    """Wrap one source in the source router."""
    router = SourceRouter({source_name: source})
    router.set_active(source_name)
    return router


def validate_runtime_compatibility(baseline: BaselineInference, runtime_channel_names: list[str] | None = None) -> None:
    """Fail early when the loaded artifact bundle cannot score the canonical runtime stream."""
    available = list(runtime_channel_names or CYTON_CHANNELS)
    incompatible: dict[str, list[str]] = {}
    for task_name, loaded_model in baseline.scorer.models.items():
        missing = [name for name in loaded_model.preprocessor.channel_names if name not in available]
        if missing:
            incompatible[task_name] = missing
    if not incompatible:
        return
    details = "; ".join(f"{task} missing {missing}" for task, missing in incompatible.items())
    raise ValueError(
        "Loaded artifacts are not compatible with the canonical 8-channel runtime stream "
        f"{available}. {details}. Train or supply a runtime-compatible artifacts bundle with --artifacts-root."
    )


def stream_outputs(
    *,
    router: SourceRouter,
    engine: DecisionEngine,
    max_chunks: int = 0,
) -> Iterator[dict[str, object]]:
    """Yield JSON-ready output payloads from the unified runtime engine."""
    chunk_count = 0
    while max_chunks <= 0 or chunk_count < max_chunks:
        chunk = router.read_chunk()
        source_name = str(chunk.metadata.get("source_name", router.active_name or "unknown"))
        outputs = engine.process_chunk(chunk)
        chunk_count += 1
        for output in outputs:
            yield {
                "timestamp_start": output.timestamp_start,
                "timestamp_end": output.timestamp_end,
                "source": source_name,
                "state": output.state,
                "concentration_raw": output.concentration_raw,
                "stress_raw": output.stress_raw,
                "concentration_smoothed": output.concentration_smoothed,
                "stress_smoothed": output.stress_smoothed,
                "confidence": output.confidence,
                "quality": output.quality,
            }


def main() -> None:
    parser = argparse.ArgumentParser(description="Stream runtime EEG decisions from synthetic, playback, or Cyton sources.")
    parser.add_argument("--source", choices=("synthetic", "playback", "cyton"), required=True, help="Active runtime source.")
    parser.add_argument("--chunks", type=int, default=0, help="Maximum chunks to read before exiting. Use 0 to stream until interrupted.")
    parser.add_argument("--warmup-seconds", type=float, default=0.0, help="Optional pre-stream warmup duration before printing decisions.")
    parser.add_argument("--chunk-seconds", type=float, default=0.5, help="Synthetic chunk duration in seconds.")
    parser.add_argument("--concentration", type=float, default=0.7, help="Synthetic concentration level in [0, 1].")
    parser.add_argument("--stress", type=float, default=0.25, help="Synthetic stress level in [0, 1].")
    parser.add_argument("--seed", type=int, default=42, help="Synthetic random seed.")
    parser.add_argument("--input", default="", help="Playback file path when --source playback.")
    parser.add_argument("--serial-port", default="", help="Cyton serial port when --source cyton, for example COM5.")
    parser.add_argument("--artifacts-root", default="", help="Optional artifacts root override.")
    parser.add_argument("--calibration-path", default="", help="Optional calibration profile path.")
    args = parser.parse_args()

    source = build_source(args)
    router = build_router(args.source, source)
    artifacts_root = Path(args.artifacts_root) if args.artifacts_root else None
    calibration_path = Path(args.calibration_path) if args.calibration_path else None
    engine = DecisionEngine(
        router=router,
        adaptation=AdaptationLayer(),
        baseline=BaselineInference(artifacts_root=artifacts_root, calibration_path=calibration_path),
        postprocessor=DecisionPostprocessor(),
    )

    try:
        validate_runtime_compatibility(engine.baseline)
        if args.warmup_seconds > 0.0:
            warmup = engine.warmup(duration_seconds=args.warmup_seconds)
            print(
                json.dumps(
                    {
                        "event": "warmup_complete",
                        "source": args.source,
                        "duration_seconds": warmup.duration_seconds,
                        "sample_rate": warmup.sample_rate,
                        "channel_names": warmup.channel_names,
                        "mean": warmup.mean.tolist(),
                        "std": warmup.std.tolist(),
                        "quality": warmup.quality,
                    }
                )
            )

        for payload in stream_outputs(router=router, engine=engine, max_chunks=args.chunks):
            print(json.dumps(payload))
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    except KeyboardInterrupt:
        pass
    finally:
        router.stop()


if __name__ == "__main__":
    main()
