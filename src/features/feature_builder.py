"""Window-to-feature conversion."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.features.bandpower import compute_bandpower_summary
from src.features.hjorth import compute_hjorth_parameters
from src.features.temporal_stats import compute_temporal_stats

ASYMMETRY_PAIRS = [
    ("AF3", "AF4"),
    ("F3", "F4"),
    ("F7", "F8"),
    ("C3", "C4"),
    ("P3", "P4"),
]


@dataclass(slots=True)
class FeatureBuilder:
    """Build deterministic flat feature vectors from EEG windows."""

    feature_names: list[str] = field(default_factory=list)
    include_absolute_bandpower: bool = False
    include_log_bandpower: bool = True
    include_relative_bandpower: bool = True
    include_temporal_stats: bool = True
    include_hjorth: bool = True
    include_ratios: bool = True
    include_asymmetry: bool = True
    ratio_feature_names: tuple[str, ...] = (
        "alpha_beta_ratio",
        "theta_beta_ratio",
        "beta_alpha_ratio",
        "alpha_theta_ratio",
        "engagement_index",
    )

    def build_window(
        self,
        window: np.ndarray,
        sampling_rate: float,
        channel_names: list[str] | None = None,
    ) -> tuple[np.ndarray, list[str]]:
        """Convert one EEG window into a flat feature vector."""
        names = channel_names or [f"ch_{idx:03d}" for idx in range(window.shape[0])]
        vector: list[float] = []
        feature_names: list[str] = []
        per_channel_relative: dict[str, dict[str, float]] = {}
        per_channel_log_power: dict[str, dict[str, float]] = {}

        for channel_name, channel in zip(names, window, strict=True):
            absolute, log_absolute, relative, ratios = compute_bandpower_summary(channel, sampling_rate)
            temporal = compute_temporal_stats(channel)
            hjorth = compute_hjorth_parameters(channel)
            per_channel_relative[channel_name] = relative
            per_channel_log_power[channel_name] = log_absolute

            if self.include_absolute_bandpower:
                for band_name, value in absolute.items():
                    feature_names.append(f"{channel_name}_bandpower_{band_name}")
                    vector.append(value)
            if self.include_log_bandpower:
                for band_name, value in log_absolute.items():
                    feature_names.append(f"{channel_name}_log_bandpower_{band_name}")
                    vector.append(value)
            if self.include_relative_bandpower:
                for band_name, value in relative.items():
                    feature_names.append(f"{channel_name}_relative_bandpower_{band_name}")
                    vector.append(value)
            if self.include_ratios:
                for name in self.ratio_feature_names:
                    value = ratios[name]
                    feature_names.append(f"{channel_name}_{name}")
                    vector.append(value)
            if self.include_temporal_stats:
                for name, value in temporal.items():
                    feature_names.append(f"{channel_name}_{name}")
                    vector.append(value)
            if self.include_hjorth:
                for name, value in hjorth.items():
                    feature_names.append(f"{channel_name}_hjorth_{name}")
                    vector.append(value)

        if self.include_asymmetry:
            for left_name, right_name in ASYMMETRY_PAIRS:
                if left_name in per_channel_relative and right_name in per_channel_relative:
                    for band_name in per_channel_relative[left_name]:
                        feature_names.append(f"{left_name}_{right_name}_asymmetry_{band_name}")
                        vector.append(per_channel_relative[left_name][band_name] - per_channel_relative[right_name][band_name])
                if left_name in per_channel_log_power and right_name in per_channel_log_power:
                    for band_name in per_channel_log_power[left_name]:
                        feature_names.append(f"{left_name}_{right_name}_log_asymmetry_{band_name}")
                        vector.append(per_channel_log_power[left_name][band_name] - per_channel_log_power[right_name][band_name])

        if not self.feature_names:
            self.feature_names = feature_names
        return np.asarray(vector, dtype=float), feature_names

    def build_matrix(
        self,
        windows: np.ndarray,
        sampling_rate: float,
        channel_names: list[str] | None = None,
    ) -> tuple[np.ndarray, list[str]]:
        """Convert a stack of windows into a feature matrix."""
        rows: list[np.ndarray] = []
        names: list[str] = list(self.feature_names)
        for window in windows:
            row, names = self.build_window(window, sampling_rate=sampling_rate, channel_names=channel_names)
            rows.append(row)
        if not rows:
            return np.empty((0, len(self.feature_names)), dtype=float), list(self.feature_names)
        return np.vstack(rows), names
