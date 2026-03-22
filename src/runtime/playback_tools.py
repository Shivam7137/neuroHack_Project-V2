"""Shared helpers for playback recording generation and export."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import time

import numpy as np

from src.generator.inference.sampler import GeneratorCondition, SyntheticSampler
from src.utils.io import ensure_dir


@dataclass(slots=True)
class PlaybackExportResult:
    """Paths to exported playback files."""

    npz_path: Path
    openbci_path: Path
    brainflow_path: Path


@dataclass(slots=True)
class PlaybackRecording:
    """Canonical in-memory synthetic recording."""

    signal: np.ndarray
    sample_rate: float
    channel_names: list[str]
    timestamps: np.ndarray
    metadata: dict[str, float | int | str]


def load_playback_npz(path: Path) -> tuple[np.ndarray, float, list[str], np.ndarray]:
    """Load a canonical playback recording."""
    payload = np.load(path, allow_pickle=True)
    signal = np.asarray(payload["signal"], dtype=float)
    sample_rate = float(payload["sampling_rate"])
    channel_names = [str(item) for item in payload["channel_names"].tolist()]
    timestamps = (
        np.asarray(payload["timestamps"], dtype=float)
        if "timestamps" in payload
        else np.arange(signal.shape[1], dtype=float) / sample_rate
    )
    return signal, sample_rate, channel_names, timestamps


def build_synthetic_recording(
    duration_sec: float = 12.0,
    concentration: float = 0.7,
    stress: float = 0.25,
    seed: int = 42,
) -> PlaybackRecording:
    """Generate a synthetic canonical recording in memory."""
    sampler = SyntheticSampler(sample_rate=250.0, random_seed=seed)
    condition = GeneratorCondition(
        concentration_level=float(concentration),
        stress_level=float(stress),
    )
    sample = sampler.sample(condition, duration_sec=float(duration_sec))
    timestamps = np.arange(sample.data.shape[1], dtype=float) / sample.sample_rate
    return PlaybackRecording(
        signal=sample.data.astype(float),
        sample_rate=float(sample.sample_rate),
        channel_names=list(sample.channel_names),
        timestamps=timestamps,
        metadata={
            "source": "synthetic_playback",
            "concentration_level": float(concentration),
            "stress_level": float(stress),
            "seed": int(seed),
            **{
                key: value
                for key, value in sample.metadata.items()
                if isinstance(value, (int, float, str))
            },
        },
    )


def save_playback_npz(recording: PlaybackRecording, output_path: Path) -> Path:
    """Persist a canonical recording as .npz."""
    ensure_dir(output_path.parent)
    np.savez(
        output_path,
        signal=recording.signal,
        sampling_rate=float(recording.sample_rate),
        channel_names=np.asarray(recording.channel_names),
        timestamps=recording.timestamps,
        **recording.metadata,
    )
    return output_path.resolve()


def generate_playback_npz(
    output_path: Path,
    duration_sec: float = 12.0,
    concentration: float = 0.7,
    stress: float = 0.25,
    seed: int = 42,
) -> Path:
    """Generate a synthetic canonical playback recording and save it as .npz."""
    recording = build_synthetic_recording(
        duration_sec=duration_sec,
        concentration=concentration,
        stress=stress,
        seed=seed,
    )
    return save_playback_npz(recording, output_path)


def build_brainflow_matrix(signal: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
    """Build a Cyton-shaped BrainFlow matrix from canonical EEG data."""
    samples = signal.shape[1]
    matrix = np.zeros((24, samples), dtype=float)
    matrix[0, :] = np.mod(np.arange(samples, dtype=float), 256.0)
    matrix[1:9, :] = signal[:8, :]
    matrix[9:12, :] = 0.0
    matrix[19:22, :] = 0.0
    matrix[22, :] = timestamps
    matrix[23, :] = 0.0
    return matrix


def write_openbci_txt(
    signal: np.ndarray,
    sample_rate: float,
    output_path: Path,
) -> Path:
    """Write a Cyton-style OpenBCI raw text file."""
    ensure_dir(output_path.parent)
    samples = signal.shape[1]
    sample_index = np.mod(np.arange(samples, dtype=int), 256).astype(int)
    accel = np.zeros((3, samples), dtype=float)
    digital = np.zeros((5, samples), dtype=int)
    analog = np.zeros((3, samples), dtype=float)
    sample_rate_text = str(int(round(sample_rate))) if float(sample_rate).is_integer() else str(sample_rate)
    start_ms = int(round(time.time() * 1000.0))
    timestamps_ms = start_ms + np.rint(np.arange(samples, dtype=float) * (1000.0 / sample_rate)).astype(np.int64)
    peak = float(np.max(np.abs(signal))) if signal.size else 1.0
    scale_to_nanovolts = 60000.0 / max(peak, 1e-6)
    exg_signal = signal * scale_to_nanovolts

    header_lines = [
        "%OpenBCI Raw EXG Data",
        f"%Number of channels = {signal.shape[0]}",
        f"%Sample Rate = {sample_rate_text} Hz",
        "%Board = OpenBCI_GUI$BoardCytonSerial",
        "Sample Index, EXG Channel 0, EXG Channel 1, EXG Channel 2, EXG Channel 3, EXG Channel 4, EXG Channel 5, EXG Channel 6, EXG Channel 7, Accel Channel 0, Accel Channel 1, Accel Channel 2, Not Used, Digital Channel 0 (D11), Digital Channel 1 (D12), Digital Channel 2 (D13), Digital Channel 3 (D17), Not Used, Digital Channel 4 (D18), Analog Channel 0, Analog Channel 1, Analog Channel 2, Timestamp, Marker Channel, Timestamp (Formatted)",
    ]
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for line in header_lines[:4]:
            handle.write(f"{line}\n")
        handle.write(header_lines[4] + "\n")
        for idx in range(samples):
            formatted_timestamp = datetime.fromtimestamp(
                timestamps_ms[idx] / 1000.0,
                tz=timezone.utc,
            ).strftime("%H:%M:%S.%f")[:-3]
            row = [str(sample_index[idx])]
            row.extend(f"{exg_signal[channel, idx]:.2f}" for channel in range(signal.shape[0]))
            row.extend(f"{accel[channel, idx]:.3f}" for channel in range(3))
            row.append("0")
            row.extend(str(int(digital[channel, idx])) for channel in range(4))
            row.append("0")
            row.append(str(int(digital[4, idx])))
            row.extend(f"{analog[channel, idx]:.0f}" for channel in range(3))
            row.append(str(int(timestamps_ms[idx])))
            row.append("0")
            row.append(formatted_timestamp)
            handle.write(", ".join(row) + "\n")
    return output_path.resolve()


def export_playback_formats(
    input_path: Path,
    output_dir: Path,
    prefix: str,
) -> PlaybackExportResult:
    """Export a canonical playback recording to OpenBCI and BrainFlow formats."""
    ensure_dir(output_dir)
    signal, sample_rate, _channel_names, timestamps = load_playback_npz(input_path)
    openbci_path = write_openbci_txt(signal, sample_rate, output_dir / f"OpenBCI-RAW-{prefix}.txt")

    from brainflow.data_filter import DataFilter

    brainflow_path = (output_dir / f"BrainFlow-RAW-{prefix}.csv").resolve()
    DataFilter.write_file(build_brainflow_matrix(signal, timestamps), str(brainflow_path), "w")
    return PlaybackExportResult(
        npz_path=input_path.resolve(),
        openbci_path=openbci_path,
        brainflow_path=brainflow_path,
    )


def export_openbci_format(
    input_path: Path,
    output_dir: Path,
    prefix: str,
) -> Path:
    """Export only the OpenBCI GUI text format."""
    ensure_dir(output_dir)
    signal, sample_rate, _channel_names, _timestamps = load_playback_npz(input_path)
    return write_openbci_txt(signal, sample_rate, output_dir / f"OpenBCI-RAW-{prefix}.txt")


def export_brainflow_format(
    input_path: Path,
    output_dir: Path,
    prefix: str,
) -> Path:
    """Export only the BrainFlow CSV format."""
    ensure_dir(output_dir)
    signal, _sample_rate, _channel_names, timestamps = load_playback_npz(input_path)
    from brainflow.data_filter import DataFilter

    brainflow_path = (output_dir / f"BrainFlow-RAW-{prefix}.csv").resolve()
    DataFilter.write_file(build_brainflow_matrix(signal, timestamps), str(brainflow_path), "w")
    return brainflow_path
