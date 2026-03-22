"""Inference API for scoring one EEG window."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from src.config import get_settings
from src.datasets.artifact_common import ARTIFACT_CLASS_NAMES
from src.features.feature_builder import FeatureBuilder
from src.models.stress_model import STRESS_CLASS_NAMES, STRESS_CLASS_TO_SCORE
from src.preprocessing.calibration import load_calibration_profile
from src.preprocessing.cleaners import apply_optional_pyprep
from src.preprocessing.normalization import demean_window_channels
from src.preprocessing.quality import WindowQualityConfig, assess_window_quality
from src.training.common import PreprocessorBundle
from src.utils.io import load_pickle


@dataclass(slots=True)
class LoadedTaskModel:
    """Pair a trained model with its preprocessor."""

    model: Any
    model_name: str
    prediction_mode: str
    decision_threshold: float | None
    preprocessor: PreprocessorBundle


@dataclass(slots=True)
class PreparedTaskInput:
    """Processed task-specific inputs reused across scoring and teacher APIs."""

    processed_window: np.ndarray
    feature_vector: np.ndarray
    feature_matrix: np.ndarray
    channel_names: list[str]
    sampling_rate: float
    calibration_applied: bool


def _expand_classifier_probabilities(
    probabilities: np.ndarray,
    observed_classes: np.ndarray,
    n_classes: int,
) -> np.ndarray:
    """Expand classifier probabilities to the full stress class space."""
    expanded = np.zeros((probabilities.shape[0], n_classes), dtype=float)
    for column_index, class_id in enumerate(observed_classes):
        expanded[:, int(class_id)] = probabilities[:, column_index]
    return expanded


def _resolve_task_artifact_dir(root: Path, task_name: str, cleanup_level: str | None) -> Path:
    """Resolve the best available artifact directory for one task."""
    if cleanup_level:
        benchmark_dir = root / "cleanup_benchmark" / cleanup_level / task_name
        if (benchmark_dir / "model.pkl").exists() and (benchmark_dir / "preprocessor.pkl").exists():
            return benchmark_dir
    direct_dir = root / task_name
    return direct_dir


def load_models(
    artifacts_root: Path | None = None,
    *,
    concentration_cleanup_level: str | None = None,
    stress_cleanup_level: str | None = None,
) -> dict[str, LoadedTaskModel]:
    """Load trained task models from disk."""
    settings = get_settings()
    root = artifacts_root or settings.artifacts_root
    models: dict[str, LoadedTaskModel] = {}
    cleanup_levels = {
        "concentration": concentration_cleanup_level if concentration_cleanup_level is not None else settings.concentration_cleanup_level,
        "stress": stress_cleanup_level if stress_cleanup_level is not None else settings.stress_cleanup_level,
    }
    for task_name in ("concentration", "stress"):
        task_root = _resolve_task_artifact_dir(root, task_name, cleanup_levels[task_name])
        model_payload = load_pickle(task_root / "model.pkl")
        preprocessor = load_pickle(task_root / "preprocessor.pkl")
        _hydrate_preprocessor(preprocessor, settings)
        models[task_name] = LoadedTaskModel(
            model=model_payload["model"],
            model_name=model_payload["model_name"],
            prediction_mode=model_payload["prediction_mode"],
            decision_threshold=model_payload.get("decision_threshold"),
            preprocessor=preprocessor,
        )
    artifact_model_path = root / "artifact" / "model.pkl"
    artifact_preprocessor_path = root / "artifact" / "preprocessor.pkl"
    if artifact_model_path.exists() and artifact_preprocessor_path.exists():
        model_payload = load_pickle(artifact_model_path)
        preprocessor = load_pickle(artifact_preprocessor_path)
        _hydrate_preprocessor(preprocessor, settings)
        models["artifact"] = LoadedTaskModel(
            model=model_payload["model"],
            model_name=model_payload["model_name"],
            prediction_mode=model_payload["prediction_mode"],
            decision_threshold=model_payload.get("decision_threshold"),
            preprocessor=preprocessor,
        )
    return models


def _hydrate_preprocessor(preprocessor: PreprocessorBundle, settings) -> None:
    """Backfill newer preprocessor fields for older serialized artifacts."""
    defaults: dict[str, object] = {
        "cleanup_level": "none",
        "apply_window_channel_demean": settings.enable_window_channel_demean,
        "quality_config": None,
        "decision_threshold": None,
        "feature_group_settings": {
            "include_log_bandpower": True,
            "include_relative_bandpower": True,
            "include_temporal_stats": True,
            "include_hjorth": True,
            "include_ratios": True,
            "include_asymmetry": True,
        },
    }
    for name, value in defaults.items():
        if not hasattr(preprocessor, name):
            setattr(preprocessor, name, value)


@lru_cache(maxsize=8)
def _cached_models(
    artifacts_root_str: str,
    concentration_cleanup_level: str | None,
    stress_cleanup_level: str | None,
) -> dict[str, LoadedTaskModel]:
    path = Path(artifacts_root_str) if artifacts_root_str else None
    return load_models(
        artifacts_root=path,
        concentration_cleanup_level=concentration_cleanup_level,
        stress_cleanup_level=stress_cleanup_level,
    )


def _load_window(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        return np.load(path).astype(float)
    if path.suffix.lower() == ".npz":
        payload = np.load(path, allow_pickle=True)
        key = "window" if "window" in payload else list(payload.keys())[0]
        return np.asarray(payload[key], dtype=float)
    if path.suffix.lower() == ".csv":
        return np.loadtxt(path, delimiter=",", dtype=float)
    raise ValueError(f"Unsupported input file format: {path.suffix}")


def _prepare_task_input(
    task: LoadedTaskModel,
    window: np.ndarray,
    sampling_rate: float,
    channel_names: list[str] | None,
    calibration_path: Path | None = None,
) -> PreparedTaskInput:
    processed, effective_sampling_rate = task.preprocessor.transform_raw_with_sampling_rate(
        window,
        sampling_rate,
        channel_names,
        apply_trim=False,
    )
    if calibration_path is not None and task.preprocessor.task_name == "artifact":
        calibration_profile = load_calibration_profile(calibration_path)
        processed = processed * calibration_profile.channel_quality_weights
    if task.preprocessor.apply_window_channel_demean:
        processed = demean_window_channels(processed)
    builder = FeatureBuilder(
        feature_names=list(task.preprocessor.feature_names),
        **task.preprocessor.feature_group_settings,
    )
    features, _ = builder.build_window(
        processed,
        sampling_rate=effective_sampling_rate,
        channel_names=task.preprocessor.channel_names,
    )
    matrix = task.preprocessor.transform_features(features.reshape(1, -1))
    return PreparedTaskInput(
        processed_window=processed,
        feature_vector=features.astype(float),
        feature_matrix=matrix,
        channel_names=list(task.preprocessor.channel_names),
        sampling_rate=effective_sampling_rate,
        calibration_applied=calibration_path is not None,
    )


def _score_task(
    task: LoadedTaskModel,
    prepared: PreparedTaskInput,
) -> dict[str, Any]:
    matrix = prepared.feature_matrix

    if task.preprocessor.task_name == "artifact":
        probabilities = task.model.predict_proba(matrix)
        probabilities = _expand_classifier_probabilities(probabilities, np.asarray(task.model.classes_), len(ARTIFACT_CLASS_NAMES))[0]
        predicted_class = int(np.argmax(probabilities))
        return {
                "quality_score": float(probabilities[0] * 100.0),
                "quality_label": "clean" if predicted_class == 0 else "noisy",
                "artifact_probabilities": {label: float(probabilities[idx]) for idx, label in enumerate(ARTIFACT_CLASS_NAMES)},
                "artifact_predicted_class": ARTIFACT_CLASS_NAMES[predicted_class],
                "calibration_applied": prepared.calibration_applied,
            }

    if task.preprocessor.task_name == "concentration":
        probabilities = task.model.predict_proba(matrix)[0]
        concentration_probability = float(probabilities[1])
        threshold = task.decision_threshold if task.decision_threshold is not None else 0.5
        return {
            "concentration_score": concentration_probability * 100.0,
            "concentration_probability": concentration_probability,
            "concentration_predicted_label": int(concentration_probability >= threshold),
        }

    if task.prediction_mode == "regressor":
        predicted_score = float(np.clip(task.model.predict(matrix)[0], 0.0, 1.0))
        class_values = np.asarray(list(STRESS_CLASS_TO_SCORE.values()))
        predicted_class = int(np.argmin(np.abs(class_values - predicted_score)))
        probabilities_dict: dict[str, float] = {}
    else:
        probabilities = task.model.predict_proba(matrix)
        probabilities = _expand_classifier_probabilities(probabilities, np.asarray(task.model.classes_), len(STRESS_CLASS_NAMES))[0]
        probabilities_dict = {label: float(probabilities[idx]) for idx, label in enumerate(STRESS_CLASS_NAMES)}
        predicted_class = int(np.argmax(probabilities))
        predicted_score = float(np.sum(probabilities * np.asarray(list(STRESS_CLASS_TO_SCORE.values()))))

    return {
        "stress_score": predicted_score * 100.0,
        "stress_class_probabilities": probabilities_dict,
        "stress_predicted_class": STRESS_CLASS_NAMES[predicted_class],
    }


class RuntimeScorer:
    """Persistent scorer that reuses loaded artifacts across many windows."""

    def __init__(
        self,
        artifacts_root: Path | None = None,
        calibration_path: Path | None = None,
        models: dict[str, LoadedTaskModel] | None = None,
        concentration_cleanup_level: str | None = None,
        stress_cleanup_level: str | None = None,
    ) -> None:
        settings = get_settings()
        root = (artifacts_root or settings.artifacts_root).resolve()
        self.settings = settings
        self.artifacts_root = root
        self.calibration_path = calibration_path.resolve() if calibration_path else None
        self.concentration_cleanup_level = concentration_cleanup_level if concentration_cleanup_level is not None else settings.concentration_cleanup_level
        self.stress_cleanup_level = stress_cleanup_level if stress_cleanup_level is not None else settings.stress_cleanup_level
        self.models = (
            models
            if models is not None
            else _cached_models(
                str(root),
                self.concentration_cleanup_level,
                self.stress_cleanup_level,
            )
        )

    def prepare_task_input(
        self,
        task_name: str,
        window: np.ndarray,
        sampling_rate: float,
        channel_names: list[str] | None = None,
    ) -> PreparedTaskInput:
        calibration_path = self.calibration_path if task_name == "artifact" else None
        return _prepare_task_input(
            self.models[task_name],
            window,
            sampling_rate,
            channel_names,
            calibration_path=calibration_path,
        )

    def extract_task_feature_embedding(
        self,
        task_name: str,
        window: np.ndarray,
        sampling_rate: float,
        channel_names: list[str] | None = None,
    ) -> np.ndarray:
        prepared = self.prepare_task_input(task_name, window, sampling_rate, channel_names)
        return prepared.feature_matrix[0].astype(float, copy=True)

    def extract_feature_embedding(
        self,
        window: np.ndarray,
        sampling_rate: float,
        channel_names: list[str] | None = None,
    ) -> np.ndarray:
        concentration = self.extract_task_feature_embedding(
            "concentration",
            window,
            sampling_rate,
            channel_names,
        )
        stress = self.extract_task_feature_embedding(
            "stress",
            window,
            sampling_rate,
            channel_names,
        )
        return np.concatenate([concentration, stress], axis=0)

    def score_task(
        self,
        task_name: str,
        window: np.ndarray,
        sampling_rate: float,
        channel_names: list[str] | None = None,
    ) -> dict[str, Any]:
        prepared = self.prepare_task_input(task_name, window, sampling_rate, channel_names)
        return _score_task(self.models[task_name], prepared)

    def score_window(
        self,
        window: np.ndarray,
        sampling_rate: float,
        channel_names: list[str] | None = None,
    ) -> dict[str, Any]:
        concentration_prepared = self.prepare_task_input("concentration", window, sampling_rate, channel_names)
        stress_prepared = self.prepare_task_input("stress", window, sampling_rate, channel_names)
        concentration = _score_task(self.models["concentration"], concentration_prepared)
        stress = _score_task(self.models["stress"], stress_prepared)
        qc_config = WindowQualityConfig(
            flat_variance_threshold=self.settings.flat_variance_threshold,
            max_abs_amplitude_threshold=self.settings.max_abs_amplitude_threshold,
            max_variance_threshold=self.settings.max_variance_threshold,
            max_line_noise_ratio=self.settings.max_line_noise_ratio,
            line_noise_frequency=self.settings.notch_freq,
        )
        qc_passed, qc_reasons = assess_window_quality(window, sampling_rate, qc_config)
        pyprep_flags = apply_optional_pyprep(
            window,
            sampling_rate=sampling_rate,
            channel_names=channel_names or [f"ch_{idx:02d}" for idx in range(window.shape[0])],
            enabled=self.settings.enable_pyprep,
        ).flags
        if "artifact" in self.models:
            artifact_prepared = self.prepare_task_input("artifact", window, sampling_rate, channel_names)
            artifact = _score_task(self.models["artifact"], artifact_prepared)
        else:
            clean_probability = 1.0 if qc_passed else 0.0
            fallback_probabilities = {label: 0.0 for label in ARTIFACT_CLASS_NAMES}
            fallback_probabilities["clean"] = clean_probability
            artifact = {
                "quality_score": clean_probability * 100.0,
                "quality_label": "clean" if qc_passed else "noisy",
                "artifact_probabilities": fallback_probabilities,
                "artifact_predicted_class": "clean" if qc_passed else "musc",
                "calibration_applied": False,
            }
        if not qc_passed:
            artifact["quality_label"] = "noisy"
        return {
            "concentration_score": concentration["concentration_score"],
            "stress_score": stress["stress_score"],
            "concentration_probability": concentration["concentration_probability"],
            "stress_class_probabilities": stress["stress_class_probabilities"],
            "stress_predicted_class": stress["stress_predicted_class"],
            "quality_score": artifact["quality_score"],
            "quality_label": artifact["quality_label"],
            "artifact_probabilities": artifact["artifact_probabilities"],
            "artifact_predicted_class": artifact["artifact_predicted_class"],
            "artifact_flags": {
                "deterministic_window_passed": qc_passed,
                "deterministic_reasons": qc_reasons,
                "pyprep_status": pyprep_flags.get("pyprep_status"),
                "pyprep_bad_channels": pyprep_flags.get("pyprep_bad_channels", []),
                "calibration_applied": artifact.get("calibration_applied", False),
            },
        }


def score_window(
    window: np.ndarray,
    sampling_rate: float,
    channel_names: list[str] | None = None,
    artifacts_root: Path | None = None,
    calibration_path: Path | None = None,
) -> dict[str, Any]:
    """Score one EEG window with both trained models."""
    scorer = RuntimeScorer(artifacts_root=artifacts_root, calibration_path=calibration_path)
    return scorer.score_window(window, sampling_rate=sampling_rate, channel_names=channel_names)


def main() -> None:
    parser = argparse.ArgumentParser(description="Score one EEG window.")
    parser.add_argument("--input", required=True, help="Path to a .npy, .npz, or .csv EEG window.")
    parser.add_argument("--sampling-rate", required=True, type=float, help="Sampling rate for the input window.")
    parser.add_argument("--channel-names", default="", help="Comma-separated channel names in input order.")
    parser.add_argument("--calibration-profile", default="", help="Optional path to a saved calibration profile.")
    args = parser.parse_args()

    channel_names = [part.strip() for part in args.channel_names.split(",") if part.strip()] or None
    window = _load_window(Path(args.input))
    calibration_path = Path(args.calibration_profile) if args.calibration_profile else None
    result = score_window(window, sampling_rate=args.sampling_rate, channel_names=channel_names, calibration_path=calibration_path)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
