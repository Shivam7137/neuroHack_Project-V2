"""Tests for playback generation and export helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.runtime.playback_tools import export_playback_formats, generate_playback_npz, load_playback_npz


def test_generate_playback_npz_and_export_formats(tmp_path: Path) -> None:
    npz_path = generate_playback_npz(
        tmp_path / "sample.npz",
        duration_sec=4.0,
        concentration=0.8,
        stress=0.2,
        seed=7,
    )
    signal, sample_rate, channel_names, timestamps = load_playback_npz(npz_path)
    assert signal.shape == (8, 1000)
    assert sample_rate == 250.0
    assert channel_names == ["Fp1", "Fp2", "C3", "C4", "P3", "P4", "O1", "O2"]
    assert len(timestamps) == 1000

    result = export_playback_formats(npz_path, tmp_path / "exports", "demo")
    assert result.openbci_path.exists()
    assert result.brainflow_path.exists()

    header = result.openbci_path.read_text(encoding="utf-8").splitlines()[:5]
    assert header[0] == "%OpenBCI Raw EXG Data"
    assert "Sample Rate = 250 Hz" in header[2]
    assert header[3] == "%Board = OpenBCI_GUI$BoardCytonSerial"
    assert header[4].startswith("Sample Index, EXG Channel 0")
