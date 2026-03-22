"""Treatment shim insertion point for runtime EEG windows."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class TreatmentResult:
    """Transformed runtime window plus quality metadata."""

    window: np.ndarray
    quality: dict[str, float]


class TreatmentShim:
    """Identity treatment used until a richer adaptive model is available."""

    def transform(
        self,
        window: np.ndarray,
        sampling_rate: float,
        channel_names: list[str],
    ) -> TreatmentResult:
        rms = float(np.sqrt(np.mean(np.square(window)))) if window.size else 0.0
        return TreatmentResult(
            window=window.astype(float, copy=False),
            quality={
                "sampling_rate": float(sampling_rate),
                "channel_count": float(len(channel_names)),
                "rms": rms,
            },
        )
