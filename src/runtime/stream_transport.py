"""Helpers for simple localhost EEG chunk streaming between terminals."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from src.inference.scorer import load_models
from src.runtime.contracts import EEGChunk

DEFAULT_STREAM_HOST = "127.0.0.1"
DEFAULT_STREAM_PORT = 8765
UDP_MAX_PAYLOAD_BYTES = 65507


def _json_scalar(value: Any) -> Any:
    """Convert runtime metadata values into JSON-safe scalars."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def encode_chunk(chunk: EEGChunk) -> str:
    """Serialize one chunk to a compact JSON line."""
    payload = {
        "timestamp_start": float(chunk.timestamp_start),
        "timestamp_end": float(chunk.timestamp_end),
        "sample_rate": float(chunk.sample_rate),
        "channel_names": list(chunk.channel_names),
        "data": np.asarray(chunk.data, dtype=float).tolist(),
        "metadata": {key: _json_scalar(value) for key, value in chunk.metadata.items()},
    }
    return json.dumps(payload, separators=(",", ":"))


def decode_chunk(payload: str) -> EEGChunk:
    """Parse one JSON line back into a runtime chunk."""
    decoded = json.loads(payload)
    return EEGChunk(
        timestamp_start=float(decoded["timestamp_start"]),
        timestamp_end=float(decoded["timestamp_end"]),
        sample_rate=float(decoded["sample_rate"]),
        channel_names=[str(name) for name in decoded["channel_names"]],
        data=np.asarray(decoded["data"], dtype=float),
        metadata={str(key): _json_scalar(value) for key, value in dict(decoded.get("metadata", {})).items()},
    )


def pack_chunk_datagram(chunk: EEGChunk, *, sequence: int = 0) -> bytes:
    """Serialize one chunk into a compact UDP-safe binary datagram."""
    data = np.asarray(chunk.data, dtype=np.float32, order="C")
    header = {
        "timestamp_start": float(chunk.timestamp_start),
        "timestamp_end": float(chunk.timestamp_end),
        "sample_rate": float(chunk.sample_rate),
        "channel_names": list(chunk.channel_names),
        "shape": list(data.shape),
        "dtype": "float32",
        "sequence": int(sequence),
        "metadata": {key: _json_scalar(value) for key, value in chunk.metadata.items()},
    }
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    packet = len(header_bytes).to_bytes(4, byteorder="big") + header_bytes + data.tobytes(order="C")
    if len(packet) > UDP_MAX_PAYLOAD_BYTES:
        raise ValueError(
            f"UDP packet is too large at {len(packet)} bytes. Reduce --chunk-seconds or channel count."
        )
    return packet


def unpack_chunk_datagram(packet: bytes) -> tuple[EEGChunk, int]:
    """Parse one UDP datagram back into a runtime chunk and sequence id."""
    if len(packet) < 4:
        raise ValueError("UDP packet is too short to contain a header.")
    header_size = int.from_bytes(packet[:4], byteorder="big")
    if len(packet) < 4 + header_size:
        raise ValueError("UDP packet ended before the declared header size.")
    header = json.loads(packet[4 : 4 + header_size].decode("utf-8"))
    data_bytes = packet[4 + header_size :]
    shape = tuple(int(value) for value in header["shape"])
    dtype = np.dtype(str(header["dtype"]))
    data = np.frombuffer(data_bytes, dtype=dtype).reshape(shape).astype(float, copy=False)
    chunk = EEGChunk(
        timestamp_start=float(header["timestamp_start"]),
        timestamp_end=float(header["timestamp_end"]),
        sample_rate=float(header["sample_rate"]),
        channel_names=[str(name) for name in header["channel_names"]],
        data=data,
        metadata={str(key): _json_scalar(value) for key, value in dict(header.get("metadata", {})).items()},
    )
    return chunk, int(header.get("sequence", 0))


def required_stream_channel_names(artifacts_root: Path | None = None) -> list[str]:
    """Return the ordered union of channels required by the saved scorers."""
    models = load_models(artifacts_root=artifacts_root)
    ordered: list[str] = []
    seen: set[str] = set()
    for task_name in ("concentration", "stress", "artifact"):
        if task_name not in models:
            continue
        for channel_name in models[task_name].preprocessor.channel_names:
            if channel_name in seen:
                continue
            ordered.append(channel_name)
            seen.add(channel_name)
    return ordered
