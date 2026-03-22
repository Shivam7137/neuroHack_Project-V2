"""Consume streamed EEG chunks and print concentration/stress scores."""

from __future__ import annotations

import argparse
import json
import socket
from pathlib import Path

from src.runtime.eeg_frame import frame_from_chunk
from src.runtime.engine import StreamingEngine
from src.runtime.stream_transport import (
    DEFAULT_STREAM_HOST,
    DEFAULT_STREAM_PORT,
    decode_chunk,
    unpack_chunk_datagram,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Connect to a streamed EEG source and score it.")
    parser.add_argument("--transport", choices=("udp", "tcp"), default="udp", help="Transport protocol.")
    parser.add_argument("--host", default=DEFAULT_STREAM_HOST, help="Bind or connect host. Defaults to localhost.")
    parser.add_argument("--port", type=int, default=DEFAULT_STREAM_PORT, help="Bind or connect port.")
    parser.add_argument("--artifacts-root", default="", help="Optional artifacts root override.")
    parser.add_argument("--calibration-path", default="", help="Optional calibration profile path.")
    args = parser.parse_args()

    artifacts_root = Path(args.artifacts_root) if args.artifacts_root else None
    calibration_path = Path(args.calibration_path) if args.calibration_path else None
    engine = StreamingEngine(artifacts_root=artifacts_root, calibration_path=calibration_path)

    if args.transport == "udp":
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as udp_socket:
            udp_socket.bind((args.host, args.port))
            print(f"listening for udp on {args.host}:{args.port}", flush=True)
            try:
                while True:
                    packet, address = udp_socket.recvfrom(65535)
                    chunk, sequence = unpack_chunk_datagram(packet)
                    frame = frame_from_chunk(chunk, source=str(chunk.metadata.get("source_name", "udp")))
                    for output in engine.process_frame(frame):
                        print(
                            json.dumps(
                                {
                                    "timestamp": output.timestamp,
                                    "source": output.source,
                                    "chunk_sequence": sequence,
                                    "sender": f"{address[0]}:{address[1]}",
                                    "concentration_score": output.concentration_score,
                                    "stress_score": output.stress_score,
                                    "concentration_probability": output.concentration_probability,
                                    "stress_predicted_class": output.stress_predicted_class,
                                }
                            ),
                            flush=True,
                        )
            except KeyboardInterrupt:
                pass
        return

    try:
        with socket.create_connection((args.host, args.port)) as connection:
            reader = connection.makefile("r", encoding="utf-8", newline="\n")
            print(f"connected to tcp {args.host}:{args.port}", flush=True)
            try:
                while True:
                    line = reader.readline()
                    if not line:
                        break
                    chunk = decode_chunk(line)
                    frame = frame_from_chunk(chunk, source=str(chunk.metadata.get("source_name", "tcp")))
                    for output in engine.process_frame(frame):
                        print(
                            json.dumps(
                                {
                                    "timestamp": output.timestamp,
                                    "source": output.source,
                                    "concentration_score": output.concentration_score,
                                    "stress_score": output.stress_score,
                                    "concentration_probability": output.concentration_probability,
                                    "stress_predicted_class": output.stress_predicted_class,
                                }
                            ),
                            flush=True,
                        )
            finally:
                reader.close()
    except ConnectionRefusedError as exc:
        raise SystemExit(f"Could not connect to generator at {args.host}:{args.port}. Start stream_generate first.") from exc
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
