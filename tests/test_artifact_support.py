from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from src.inference.scorer import score_window
from src.preprocessing.calibration import build_calibration_profile, load_calibration_profile, save_calibration_profile
from src.preprocessing.cleaners import apply_optional_autoreject, apply_optional_pyprep


def test_optional_cleaners_skip_cleanly_without_backends(monkeypatch) -> None:
    signal = np.ones((4, 128), dtype=float)
    windows = np.ones((4, 4, 128), dtype=float)
    monkeypatch.setattr("src.preprocessing.cleaners._load_pyprep_backend", lambda: None)
    monkeypatch.setattr("src.preprocessing.cleaners._load_autoreject_backend", lambda: None)

    pyprep_result = apply_optional_pyprep(signal, sampling_rate=128.0, channel_names=["C3", "C4", "P3", "P4"], enabled=True)
    autoreject_result = apply_optional_autoreject(windows, sampling_rate=128.0, channel_names=["C3", "C4", "P3", "P4"], enabled=True)

    assert pyprep_result.flags["pyprep_status"] == "unavailable"
    assert autoreject_result.flags["autoreject_status"] == "unavailable"
    assert autoreject_result.windows.shape == windows.shape


def test_optional_cleaners_use_fake_backends(monkeypatch) -> None:
    signal = np.vstack([np.linspace(-1.0, 1.0, 128) for _ in range(4)])
    windows = np.stack([signal, signal * 0.5, signal * 0.2, signal * 3.0], axis=0)

    class FakeRawArray:
        def __init__(self, data, info, verbose="ERROR"):
            self.data = data
            self.info = info

    class FakeEpochsArray:
        def __init__(self, data, info, verbose="ERROR"):
            self.data = data
            self.info = info

    class FakeNoisyChannels:
        def __init__(self, raw, do_detrend=False, random_state=42):
            self.raw = raw

        def find_all_bads(self) -> None:
            return None

        def get_bads(self):
            return ["C4"]

    fake_mne = SimpleNamespace(
        create_info=lambda ch_names, sfreq, ch_types: {"ch_names": ch_names, "sfreq": sfreq, "ch_types": ch_types},
        io=SimpleNamespace(RawArray=FakeRawArray),
        EpochsArray=FakeEpochsArray,
    )
    monkeypatch.setattr("src.preprocessing.cleaners._load_pyprep_backend", lambda: (fake_mne, FakeNoisyChannels))
    monkeypatch.setattr(
        "src.preprocessing.cleaners._load_autoreject_backend",
        lambda: (fake_mne, lambda epochs, verbose=False: {"eeg": 4.0}),
    )

    pyprep_result = apply_optional_pyprep(signal, sampling_rate=128.0, channel_names=["C3", "C4", "P3", "P4"], enabled=True)
    autoreject_result = apply_optional_autoreject(windows, sampling_rate=128.0, channel_names=["C3", "C4", "P3", "P4"], enabled=True)

    assert pyprep_result.flags["pyprep_used"] is True
    assert pyprep_result.flags["pyprep_bad_channels"] == ["C4"]
    assert autoreject_result.flags["autoreject_used"] is True
    assert autoreject_result.flags["autoreject_rejected_windows"] == 1
    assert autoreject_result.windows.shape[0] == 3


def test_calibration_profile_round_trip_and_inference(trained_artifacts: dict[str, Path], tmp_path: Path) -> None:
    payload = np.load(trained_artifacts["tuar_root"] / "artifact_subj_00_clean.npz", allow_pickle=True)
    signal = np.asarray(payload["signal"], dtype=float)
    channel_names = [str(item) for item in payload["channel_names"].tolist()]
    profile = build_calibration_profile(
        signal,
        sampling_rate=float(payload["sampling_rate"]),
        channel_names=channel_names,
        subject_id="artifact_subj_00",
        session_id="artifact_subj_00_clean",
    )
    calibration_path = tmp_path / "artifact_calibration.pkl"
    save_calibration_profile(profile, calibration_path)
    loaded_profile = load_calibration_profile(calibration_path)

    assert loaded_profile.subject_id == "artifact_subj_00"

    result = score_window(
        signal,
        sampling_rate=float(payload["sampling_rate"]),
        channel_names=channel_names,
        artifacts_root=trained_artifacts["artifacts_root"],
        calibration_path=calibration_path,
    )

    assert result["artifact_flags"]["calibration_applied"] is True
    assert "quality_score" in result
