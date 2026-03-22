"""Project configuration helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path


def _env_path(name: str, default: str) -> Path:
    return Path(os.getenv(name, default)).expanduser().resolve()


def _default_eegmat_root() -> Path:
    env_value = os.getenv("EEGMAT_ROOT")
    if env_value:
        return Path(env_value).expanduser().resolve()
    preferred = Path("./data/eegmat").resolve()
    extracted = Path("./data/eeg-during-mental-arithmetic-tasks-1.0.0").resolve()
    return extracted if extracted.exists() else preferred


def _default_stress_root() -> Path:
    env_value = os.getenv("STRESS_DATA_ROOT")
    if env_value:
        return Path(env_value).expanduser().resolve()
    preferred = Path("./data/stress").resolve()
    stew = Path("./data/STEW Dataset").resolve()
    return stew if stew.exists() else preferred


def _default_tuar_root() -> Path:
    env_value = os.getenv("TUAR_ROOT")
    if env_value:
        return Path(env_value).expanduser().resolve()
    return Path("./data/tuar").resolve()


def _default_eegdenoisenet_root() -> Path:
    env_value = os.getenv("EEGDENOISENET_ROOT")
    if env_value:
        return Path(env_value).expanduser().resolve()
    return Path("./data/eegdenoisenet").resolve()


def _env_float(name: str, default: float | None) -> float | None:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return float(value)


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return default if value is None or value == "" else int(value)


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_csv(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return tuple(part.strip() for part in value.split(",") if part.strip())


@dataclass(slots=True)
class Settings:
    """Typed runtime settings."""

    data_root: Path = field(default_factory=lambda: _env_path("DATA_ROOT", "./data"))
    eegmat_root: Path = field(default_factory=_default_eegmat_root)
    stress_data_root: Path = field(default_factory=_default_stress_root)
    tuar_root: Path = field(default_factory=_default_tuar_root)
    eegdenoisenet_root: Path = field(default_factory=_default_eegdenoisenet_root)
    artifacts_root: Path = field(default_factory=lambda: _env_path("ARTIFACTS_ROOT", "./artifacts"))
    window_seconds: float = field(default_factory=lambda: _env_float("WINDOW_SECONDS", 2.0) or 2.0)
    stride_seconds: float = field(default_factory=lambda: _env_float("STRIDE_SECONDS", 0.5) or 0.5)
    bandpass_low: float | None = field(default_factory=lambda: _env_float("BANDPASS_LOW", 1.0))
    bandpass_high: float | None = field(default_factory=lambda: _env_float("BANDPASS_HIGH", 40.0))
    notch_freq: float | None = field(default_factory=lambda: _env_float("NOTCH_FREQ", 50.0))
    random_seed: int = field(default_factory=lambda: _env_int("RANDOM_SEED", 42))
    auto_download_eegmat: bool = field(default_factory=lambda: _env_bool("AUTO_DOWNLOAD_EEGMAT", True))
    enable_pyprep: bool = field(default_factory=lambda: _env_bool("ENABLE_PYPREP", False))
    enable_autoreject: bool = field(default_factory=lambda: _env_bool("ENABLE_AUTOREJECT", False))
    cleanup_level: str = os.getenv("CLEANUP_LEVEL", "none").strip().lower() or "none"
    concentration_cleanup_level: str = os.getenv("CONCENTRATION_CLEANUP_LEVEL", "light").strip().lower() or "light"
    stress_cleanup_level: str = os.getenv("STRESS_CLEANUP_LEVEL", "none").strip().lower() or "none"
    artifact_cleanup_level: str = os.getenv("ARTIFACT_CLEANUP_LEVEL", "none").strip().lower() or "none"
    preprocessing_profile: str = os.getenv("PREPROCESSING_PROFILE", "auto")
    split_train: float = 0.70
    split_val: float = 0.15
    split_test: float = 0.15
    stress_default_sampling_rate: float = 128.0
    eegmat_target_sampling_rate: float = 128.0
    stew_trim_seconds: float = 15.0
    enable_window_channel_demean: bool = True
    enable_window_quality_control: bool = True
    flat_variance_threshold: float = 1e-4
    max_abs_amplitude_threshold: float = 20.0
    max_variance_threshold: float = 50.0
    max_line_noise_ratio: float = 0.35
    supported_stress_suffixes: tuple[str, ...] = (".npy", ".npz", ".csv", ".edf", ".txt")
    concentration_candidates: tuple[str, ...] = ("logistic_regression", "linear_svm", "rbf_svm", "random_forest")
    stress_candidates: tuple[str, ...] = ("svr_rbf", "stress_regressor_random_forest", "stress_regressor_hist_gradient_boosting")
    stress_classifier_candidates: tuple[str, ...] = ("multinomial_logistic_regression", "linear_svm", "rbf_svm", "random_forest_classifier")
    artifact_candidates: tuple[str, ...] = ("multinomial_logistic_regression", "linear_svm", "random_forest_classifier")
    cyton_serial_port: str = os.getenv("CYTON_SERIAL_PORT", "").strip()
    cyton_channel_names: tuple[str, ...] = field(
        default_factory=lambda: _env_csv(
            "CYTON_CHANNEL_NAMES",
            ("Fp1", "Fp2", "C3", "C4", "P3", "P4", "O1", "O2"),
        )
    )
    runtime_chunk_seconds: float = field(default_factory=lambda: _env_float("RUNTIME_CHUNK_SECONDS", 0.5) or 0.5)
    runtime_window_seconds: float = field(default_factory=lambda: _env_float("RUNTIME_WINDOW_SECONDS", 2.0) or 2.0)
    runtime_stride_seconds: float = field(default_factory=lambda: _env_float("RUNTIME_STRIDE_SECONDS", 0.25) or 0.25)
    runtime_buffer_seconds: float = field(default_factory=lambda: _env_float("RUNTIME_BUFFER_SECONDS", 10.0) or 10.0)

    def ensure_roots(self) -> None:
        """Create writable roots used by the pipeline."""
        self.data_root.mkdir(parents=True, exist_ok=True)
        self.eegmat_root.mkdir(parents=True, exist_ok=True)
        self.stress_data_root.mkdir(parents=True, exist_ok=True)
        self.tuar_root.mkdir(parents=True, exist_ok=True)
        self.eegdenoisenet_root.mkdir(parents=True, exist_ok=True)
        self.artifacts_root.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached settings."""
    settings = Settings()
    settings.ensure_roots()
    return settings
