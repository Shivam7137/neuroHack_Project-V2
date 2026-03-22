"""Deterministic EEG signal filters."""

from __future__ import annotations

import numpy as np
from scipy import signal

from src.preprocessing.cleanup import apply_cleanup_treatment


def detrend_channels(eeg: np.ndarray) -> np.ndarray:
    """Detrend each EEG channel independently."""
    return signal.detrend(eeg, axis=-1, type="linear")


def bandpass_filter(eeg: np.ndarray, sampling_rate: float, low: float | None, high: float | None) -> np.ndarray:
    """Apply an optional Butterworth bandpass filter."""
    if low is None and high is None:
        return eeg
    nyquist = 0.5 * sampling_rate
    low_norm = None if low is None else low / nyquist
    high_norm = None if high is None else high / nyquist
    if low_norm is not None and high_norm is not None:
        btype = "bandpass"
        wn = [low_norm, high_norm]
    elif low_norm is not None:
        btype = "highpass"
        wn = low_norm
    else:
        btype = "lowpass"
        wn = high_norm
    sos = signal.butter(N=4, Wn=wn, btype=btype, output="sos")
    return signal.sosfiltfilt(sos, eeg, axis=-1)


def notch_filter(eeg: np.ndarray, sampling_rate: float, notch_freq: float | None, quality: float = 30.0) -> np.ndarray:
    """Apply an optional notch filter."""
    if notch_freq is None or notch_freq <= 0:
        return eeg
    nyquist = 0.5 * sampling_rate
    norm_freq = notch_freq / nyquist
    if norm_freq >= 1.0:
        return eeg
    b_coeff, a_coeff = signal.iirnotch(norm_freq, quality)
    return signal.filtfilt(b_coeff, a_coeff, eeg, axis=-1)


def preprocess_signal(
    eeg: np.ndarray,
    sampling_rate: float,
    bandpass_low: float | None,
    bandpass_high: float | None,
    notch_freq: float | None,
    cleanup_level: str = "none",
) -> np.ndarray:
    """Run the deterministic preprocessing stack."""
    processed = detrend_channels(eeg)
    processed = bandpass_filter(processed, sampling_rate, bandpass_low, bandpass_high)
    processed = notch_filter(processed, sampling_rate, notch_freq)
    processed = apply_cleanup_treatment(processed, sampling_rate=sampling_rate, cleanup_level=cleanup_level)
    return processed
