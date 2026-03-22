"""Serve synthetic EEG chunks over localhost for a separate scoring process."""

from __future__ import annotations

import argparse
import socket
import time
from pathlib import Path

from src.runtime.sources.synthetic_source import SyntheticSource
from src.runtime.stream_transport import (
    DEFAULT_STREAM_HOST,
    DEFAULT_STREAM_PORT,
    encode_chunk,
    pack_chunk_datagram,
    required_stream_channel_names,
)


def _resolve_channel_names(args: argparse.Namespace) -> list[str]:
    if args.channels.strip():
        return [item.strip() for item in args.channels.split(",") if item.strip()]
    artifacts_root = Path(args.artifacts_root) if args.artifacts_root else None
    return required_stream_channel_names(artifacts_root=artifacts_root)


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve synthetic EEG chunks over UDP or TCP.")
    parser.add_argument("--transport", choices=("udp", "tcp"), default="udp", help="Transport protocol.")
    parser.add_argument("--host", default=DEFAULT_STREAM_HOST, help="Bind host. Defaults to localhost.")
    parser.add_argument("--port", type=int, default=DEFAULT_STREAM_PORT, help="Bind TCP port.")
    parser.add_argument("--chunks", type=int, default=0, help="Maximum chunks to send. Use 0 to stream until interrupted.")
    parser.add_argument("--chunk-seconds", type=float, default=0.5, help="Chunk duration in seconds.")
    parser.add_argument("--concentration", type=float, default=0.7, help="Synthetic concentration level in [0, 1].")
    parser.add_argument("--stress", type=float, default=0.25, help="Synthetic stress level in [0, 1].")
    parser.add_argument("--seed", type=int, default=42, help="Synthetic random seed.")
    parser.add_argument("--artifacts-root", default="", help="Optional artifacts root used to derive required channels.")
    parser.add_argument("--channels", default="", help="Optional comma-separated channel override.")
    args = parser.parse_args()

    channel_names = _resolve_channel_names(args)
    source = SyntheticSource(chunk_seconds=args.chunk_seconds, channel_names=channel_names)
    source.set_condition(args.concentration, args.stress)
    source.set_seed(args.seed)

    if args.transport == "udp":
        print(
            f"sending udp to {args.host}:{args.port} with {len(channel_names)} channels: "
            + ", ".join(channel_names),
            flush=True,
        )
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as udp_socket:
            source.start()
            sent = 0
            try:
                while args.chunks <= 0 or sent < args.chunks:
                    chunk = source.read_chunk()
                    chunk.metadata.setdefault("source_name", source.source_name)
                    udp_socket.sendto(pack_chunk_datagram(chunk, sequence=sent), (args.host, args.port))
                    sent += 1
                    time.sleep(max(0.0, chunk.timestamp_end - chunk.timestamp_start))
            except KeyboardInterrupt:
                pass
            finally:
                source.stop()
        return

    with socket.create_server((args.host, args.port), reuse_port=False) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        print(
            f"listening for tcp on {args.host}:{args.port} with {len(channel_names)} channels: "
            + ", ".join(channel_names),
            flush=True,
        )
        connection, address = server.accept()
        with connection:
            print(f"classifier connected from {address[0]}:{address[1]}", flush=True)
            writer = connection.makefile("w", encoding="utf-8", newline="\n")
            source.start()
            sent = 0
            try:
                while args.chunks <= 0 or sent < args.chunks:
                    chunk = source.read_chunk()
                    chunk.metadata.setdefault("source_name", source.source_name)
                    writer.write(encode_chunk(chunk))
                    writer.write("\n")
                    writer.flush()
                    sent += 1
                    time.sleep(max(0.0, chunk.timestamp_end - chunk.timestamp_start))
            except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
                pass
            except KeyboardInterrupt:
                pass
            finally:
                source.stop()
                try:
                    writer.close()
                except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
                    pass


if __name__ == "__main__":
    main()
