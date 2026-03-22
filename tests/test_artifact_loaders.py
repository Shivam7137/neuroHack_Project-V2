from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import Settings
from src.datasets.artifact_tuar_loader import TUARLoader
from src.datasets.eegdenoisenet_loader import EEGdenoiseNetLoader, build_synthetic_artifact_bundle


def _settings_for_roots(tmp_root: Path, tuar_root: Path, eegdenoisenet_root: Path) -> Settings:
    return Settings(
        data_root=tmp_root / "data",
        eegmat_root=tmp_root / "eegmat",
        stress_data_root=tmp_root / "stress",
        tuar_root=tuar_root,
        eegdenoisenet_root=eegdenoisenet_root,
        artifacts_root=tmp_root / "artifacts",
        auto_download_eegmat=False,
    )


def test_tuar_loader_loads_manifest(synthetic_roots: dict[str, Path], tmp_path: Path) -> None:
    settings = _settings_for_roots(tmp_path, synthetic_roots["tuar_root"], synthetic_roots["eegdenoisenet_root"])
    loader = TUARLoader(settings=settings)

    bundle = loader.load_raw()

    assert len(bundle.recordings) == 48
    assert set(bundle.metadata["raw_label"]) == {"clean", "eyem", "chew", "shiv", "elpp", "musc"}
    assert set(bundle.metadata["mapped_label"]) == {0, 1, 2, 3, 4, 5}


def test_eegdenoisenet_loader_loads_epochs_and_synthetic_bundle(synthetic_roots: dict[str, Path], tmp_path: Path) -> None:
    settings = _settings_for_roots(tmp_path, synthetic_roots["tuar_root"], synthetic_roots["eegdenoisenet_root"])
    loader = EEGdenoiseNetLoader(settings=settings)

    epochs = loader.load_epochs()
    bundle = build_synthetic_artifact_bundle(epochs)

    assert epochs is not None
    assert len(epochs.clean_epochs) == 8
    assert len(epochs.eog_epochs) == 8
    assert len(epochs.emg_epochs) == 8
    assert bundle is not None
    assert {"clean", "eyem", "musc"} <= set(bundle.metadata["raw_label"])


def test_tuar_loader_raises_for_missing_recording(tmp_path: Path) -> None:
    tuar_root = tmp_path / "tuar"
    tuar_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "file_path": "missing_file.npz",
                "subject_id": "artifact_subj_00",
                "session_id": "artifact_subj_00_clean",
                "label": "clean",
            }
        ]
    ).to_csv(tuar_root / "manifest.csv", index=False)

    settings = _settings_for_roots(tmp_path, tuar_root, tmp_path / "eegdenoisenet")
    loader = TUARLoader(settings=settings)

    with pytest.raises(FileNotFoundError):
        loader.load_raw()
