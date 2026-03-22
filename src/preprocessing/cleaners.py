"""Optional integrations for artifact-oriented cleaning baselines."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(slots=True)
class CleanerResult:
    """Signal plus metadata emitted by optional cleaning stages."""

    signal: np.ndarray
    flags: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class WindowCleanerResult:
    """Window stack plus rejection metadata emitted by optional cleaners."""

    windows: np.ndarray
    keep_mask: np.ndarray
    flags: dict[str, Any] = field(default_factory=dict)


def _load_pyprep_backend() -> tuple[Any, Any] | None:
    try:
        import mne  # type: ignore
        from pyprep.find_noisy_channels import NoisyChannels  # type: ignore
    except ImportError:
        return None
    return mne, NoisyChannels


def _load_autoreject_backend() -> tuple[Any, Any] | None:
    try:
        import mne  # type: ignore
        from autoreject import get_rejection_threshold  # type: ignore
    except ImportError:
        return None
    return mne, get_rejection_threshold


def apply_optional_pyprep(
    signal: np.ndarray,
    *,
    sampling_rate: float,
    channel_names: list[str],
    enabled: bool,
) -> CleanerResult:
    """Use PyPREP noisy-channel detection when available, otherwise no-op."""
    flags: dict[str, Any] = {
        "pyprep_enabled": bool(enabled),
        "pyprep_used": False,
        "pyprep_status": "disabled",
        "pyprep_bad_channels": [],
    }
    if not enabled:
        return CleanerResult(signal=signal, flags=flags)

    backend = _load_pyprep_backend()
    if backend is None:
        flags["pyprep_status"] = "unavailable"
        return CleanerResult(signal=signal, flags=flags)

    mne, noisy_channels_cls = backend
    try:
        info = mne.create_info(ch_names=channel_names, sfreq=float(sampling_rate), ch_types=["eeg"] * len(channel_names))
        raw = mne.io.RawArray(signal, info, verbose="ERROR")
        noisy = noisy_channels_cls(raw, do_detrend=False, random_state=42)
        noisy.find_all_bads()
        bad_channels = list(noisy.get_bads())
        bad_index = {name: idx for idx, name in enumerate(channel_names) if name in bad_channels}
        if bad_index and len(bad_index) < len(channel_names):
            good_indices = [idx for idx, name in enumerate(channel_names) if name not in bad_channels]
            cleaned = signal - np.mean(signal[good_indices], axis=0, keepdims=True)
        else:
            cleaned = signal
        flags.update(
            {
                "pyprep_used": True,
                "pyprep_status": "ok",
                "pyprep_bad_channels": bad_channels,
            }
        )
        return CleanerResult(signal=cleaned, flags=flags)
    except Exception as exc:  # pragma: no cover - defensive fallback around optional deps
        flags["pyprep_status"] = f"error:{type(exc).__name__}"
        return CleanerResult(signal=signal, flags=flags)


def apply_optional_autoreject(
    windows: np.ndarray,
    *,
    sampling_rate: float,
    channel_names: list[str],
    enabled: bool,
) -> WindowCleanerResult:
    """Use AutoReject rejection thresholds when available, otherwise no-op."""
    keep_mask = np.ones((len(windows),), dtype=bool)
    flags: dict[str, Any] = {
        "autoreject_enabled": bool(enabled),
        "autoreject_used": False,
        "autoreject_status": "disabled",
        "autoreject_rejected_windows": 0,
    }
    if not enabled:
        return WindowCleanerResult(windows=windows, keep_mask=keep_mask, flags=flags)
    if len(windows) < 4:
        flags["autoreject_status"] = "insufficient_windows"
        return WindowCleanerResult(windows=windows, keep_mask=keep_mask, flags=flags)

    backend = _load_autoreject_backend()
    if backend is None:
        flags["autoreject_status"] = "unavailable"
        return WindowCleanerResult(windows=windows, keep_mask=keep_mask, flags=flags)

    mne, get_rejection_threshold = backend
    try:
        info = mne.create_info(ch_names=channel_names, sfreq=float(sampling_rate), ch_types=["eeg"] * len(channel_names))
        epochs = mne.EpochsArray(windows, info, verbose="ERROR")
        thresholds = get_rejection_threshold(epochs, verbose=False)
        eeg_threshold = float(thresholds.get("eeg")) if thresholds.get("eeg") is not None else None
        if eeg_threshold is None:
            flags["autoreject_status"] = "no_threshold"
            return WindowCleanerResult(windows=windows, keep_mask=keep_mask, flags=flags)
        peak_to_peak = np.ptp(windows, axis=-1).max(axis=1)
        keep_mask = peak_to_peak <= eeg_threshold
        flags.update(
            {
                "autoreject_used": True,
                "autoreject_status": "ok",
                "autoreject_threshold": eeg_threshold,
                "autoreject_rejected_windows": int(np.sum(~keep_mask)),
            }
        )
        return WindowCleanerResult(windows=windows[keep_mask], keep_mask=keep_mask, flags=flags)
    except Exception as exc:  # pragma: no cover - defensive fallback around optional deps
        flags["autoreject_status"] = f"error:{type(exc).__name__}"
        return WindowCleanerResult(windows=windows, keep_mask=keep_mask, flags=flags)
