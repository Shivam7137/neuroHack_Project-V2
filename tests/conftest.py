"""Shared pytest fixtures for synthetic EEG data."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _make_signal(class_id: int, subject_offset: float, channels: int = 4, samples: int = 768, sampling_rate: int = 128) -> np.ndarray:
    time = np.arange(samples) / sampling_rate
    signal_rows = []
    for channel in range(channels):
        phase = subject_offset + channel * 0.2
        alpha = np.sin(2 * np.pi * 10 * time + phase)
        beta = np.sin(2 * np.pi * 20 * time + phase)
        theta = np.sin(2 * np.pi * 6 * time + phase)
        if class_id == 0:
            row = 1.2 * alpha + 0.2 * beta + 0.1 * theta
        elif class_id == 1:
            row = 0.8 * alpha + 0.8 * beta + 0.15 * theta
        elif class_id == 2:
            row = 0.4 * alpha + 1.2 * beta + 0.2 * theta
        else:
            row = 0.2 * alpha + 1.6 * beta + 0.3 * theta
        signal_rows.append(row + 0.02 * (channel + 1))
    return np.vstack(signal_rows).astype(float)


def _make_artifact_signal(label: str, subject_offset: float, channels: int = 4, samples: int = 768, sampling_rate: int = 128) -> np.ndarray:
    time = np.arange(samples) / sampling_rate
    clean = _make_signal(1, subject_offset, channels=channels, samples=samples, sampling_rate=sampling_rate)
    if label == "clean":
        return clean
    if label == "eyem":
        blink = 2.5 * np.sin(2 * np.pi * 1.2 * time)
        return clean + blink
    if label == "chew":
        burst = 0.8 * np.sign(np.sin(2 * np.pi * 2.5 * time))
        return clean + burst
    if label == "shiv":
        tremor = 1.2 * np.sin(2 * np.pi * 7.0 * time)
        return clean + tremor
    if label == "elpp":
        spikes = clean.copy()
        spikes[:, 96:104] += 5.0
        spikes[:, 288:296] -= 4.5
        return spikes
    if label == "musc":
        muscle = 0.9 * np.sin(2 * np.pi * 35.0 * time + subject_offset)
        return clean + muscle
    raise ValueError(f"Unsupported artifact label fixture: {label}")


def _write_eegmat_fixture(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "Subject": [f"Subject{idx:02d}" for idx in range(8)],
            "Age": [20 + idx for idx in range(8)],
            "Gender": ["F", "M"] * 4,
            "Recording year": [2011] * 8,
            "Number of subtractions": [10.0 + idx for idx in range(8)],
            "Count quality": [idx % 2 for idx in range(8)],
        }
    ).to_csv(root / "subject-info.csv", index=False)
    channel_names = np.asarray(["C3", "C4", "P3", "P4"])
    for subject_idx in range(8):
        subject = f"Subject{subject_idx:02d}"
        baseline = _make_signal(0, subject_idx * 0.1)
        concentration = _make_signal(3, subject_idx * 0.1)
        np.savez(root / f"{subject}_1.npz", signal=baseline, sampling_rate=128, channel_names=channel_names)
        np.savez(root / f"{subject}_2.npz", signal=concentration, sampling_rate=128, channel_names=channel_names)


def _write_stress_fixture(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    channel_names = json_channel_names = '["C3", "C4", "P3", "P4"]'
    labels = ["natural", "low", "mid", "high"]
    for subject_idx in range(8):
        subject = f"stress_subj_{subject_idx:02d}"
        subject_dir = root / subject
        subject_dir.mkdir(parents=True, exist_ok=True)
        for class_id, label in enumerate(labels):
            signal = _make_signal(class_id, subject_idx * 0.13)
            file_path = subject_dir / f"{label}.npz"
            np.savez(file_path, signal=signal, sampling_rate=128, channel_names=np.asarray(["C3", "C4", "P3", "P4"]))
            rows.append(
                {
                    "file_path": str(file_path.relative_to(root)),
                    "subject_id": subject,
                    "session_id": f"{subject}_{label}",
                    "label": label,
                    "sampling_rate": 128,
                    "channel_names": json_channel_names,
                    "target_score": { "natural": 0.0, "low": 0.33, "mid": 0.66, "high": 1.0 }[label],
                }
            )
    pd.DataFrame(rows).to_csv(root / "manifest.csv", index=False)


def _write_stew_fixture(root: Path, include_ratings: bool = True) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for subject_idx in range(8):
        subject_id = f"sub{subject_idx + 1:02d}"
        low_signal = _make_signal(0, subject_idx * 0.07, channels=14, samples=1280)
        high_signal = _make_signal(3, subject_idx * 0.07, channels=14, samples=1280)
        np.savetxt(root / f"{subject_id}_lo.txt", low_signal.T, fmt="%.8f")
        np.savetxt(root / f"{subject_id}_hi.txt", high_signal.T, fmt="%.8f")
    if include_ratings:
        lines = [f"{idx + 1}, {2 + (idx % 2)}, {7 + (idx % 3 == 0)}" for idx in range(8)]
        (root / "ratings.txt").write_text("\n".join(lines), encoding="utf-8")


def _write_tuar_fixture(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    channel_names = np.asarray(["C3", "C4", "P3", "P4"])
    rows = []
    labels = ["clean", "eyem", "chew", "shiv", "elpp", "musc"]
    for subject_idx in range(8):
        subject = f"artifact_subj_{subject_idx:02d}"
        for label in labels:
            signal = _make_artifact_signal(label, subject_idx * 0.09)
            file_path = root / f"{subject}_{label}.npz"
            np.savez(file_path, signal=signal, sampling_rate=128, channel_names=channel_names)
            rows.append(
                {
                    "file_path": str(file_path.relative_to(root)),
                    "subject_id": subject,
                    "session_id": f"{subject}_{label}",
                    "label": label,
                    "sampling_rate": 128,
                    "channel_names": '["C3", "C4", "P3", "P4"]',
                }
            )
    pd.DataFrame(rows).to_csv(root / "manifest.csv", index=False)


def _write_eegdenoisenet_fixture(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    channel_names = np.asarray(["C3", "C4", "P3", "P4"])
    for index in range(8):
        clean = _make_signal(0, index * 0.05)
        eog = _make_artifact_signal("eyem", index * 0.05) - clean
        emg = _make_artifact_signal("musc", index * 0.05) - clean
        payloads = {"clean": clean, "eog": eog, "emg": emg}
        for sample_type, signal in payloads.items():
            file_path = root / f"{sample_type}_{index:02d}.npz"
            np.savez(file_path, signal=signal, sampling_rate=128, channel_names=channel_names)
            rows.append(
                {
                    "file_path": str(file_path.relative_to(root)),
                    "sample_type": sample_type,
                    "subject_id": f"{sample_type}_{index:02d}",
                    "sampling_rate": 128,
                    "channel_names": '["C3", "C4", "P3", "P4"]',
                }
            )
    pd.DataFrame(rows).to_csv(root / "manifest.csv", index=False)


def _run_cli(args: list[str], env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *args],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )


@pytest.fixture()
def synthetic_roots(tmp_path: Path) -> dict[str, Path]:
    data_root = tmp_path / "data"
    eegmat_root = data_root / "eegmat"
    stress_root = data_root / "stress"
    tuar_root = data_root / "tuar"
    eegdenoisenet_root = data_root / "eegdenoisenet"
    artifacts_root = tmp_path / "artifacts"
    _write_eegmat_fixture(eegmat_root)
    _write_stress_fixture(stress_root)
    _write_tuar_fixture(tuar_root)
    _write_eegdenoisenet_fixture(eegdenoisenet_root)
    return {
        "data_root": data_root,
        "eegmat_root": eegmat_root,
        "stress_root": stress_root,
        "tuar_root": tuar_root,
        "eegdenoisenet_root": eegdenoisenet_root,
        "artifacts_root": artifacts_root,
    }


@pytest.fixture()
def trained_artifacts(synthetic_roots: dict[str, Path]) -> dict[str, Path]:
    env = os.environ.copy()
    env.update(
        {
            "DATA_ROOT": str(synthetic_roots["data_root"]),
            "EEGMAT_ROOT": str(synthetic_roots["eegmat_root"]),
            "STRESS_DATA_ROOT": str(synthetic_roots["stress_root"]),
            "TUAR_ROOT": str(synthetic_roots["tuar_root"]),
            "EEGDENOISENET_ROOT": str(synthetic_roots["eegdenoisenet_root"]),
            "ARTIFACTS_ROOT": str(synthetic_roots["artifacts_root"]),
            "WINDOW_SECONDS": "2.0",
            "STRIDE_SECONDS": "1.0",
            "BANDPASS_LOW": "1.0",
            "BANDPASS_HIGH": "30.0",
            "NOTCH_FREQ": "",
            "RANDOM_SEED": "7",
            "AUTO_DOWNLOAD_EEGMAT": "false",
            "ENABLE_PYPREP": "false",
            "ENABLE_AUTOREJECT": "false",
        }
    )

    _run_cli(["-m", "src.training.train_concentration"], env)
    _run_cli(["-m", "src.training.train_stress"], env)
    _run_cli(["-m", "src.training.train_artifact"], env)
    _run_cli(["-m", "src.evaluation.evaluate_all"], env)

    return {
        **synthetic_roots,
        "env": env,
    }


@pytest.fixture()
def synthetic_stew_root(tmp_path: Path) -> Path:
    stew_root = tmp_path / "stew"
    _write_stew_fixture(stew_root)
    return stew_root
