"""Baseline estimator factory helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC, SVR


@dataclass(slots=True)
class ModelSpec:
    """Description of one trainable model candidate."""

    name: str
    build: Callable[[], Any]
    requires_scaling: bool
    prediction_mode: str = "classifier"


def get_model_spec(name: str) -> ModelSpec:
    """Return a configured model spec by name."""
    specs: dict[str, ModelSpec] = {
        "logistic_regression": ModelSpec(
            name="logistic_regression",
            build=lambda: LogisticRegression(max_iter=1000, solver="liblinear", class_weight="balanced"),
            requires_scaling=True,
        ),
        "multinomial_logistic_regression": ModelSpec(
            name="multinomial_logistic_regression",
            build=lambda: LogisticRegression(
                max_iter=2000,
                solver="lbfgs",
                class_weight="balanced",
            ),
            requires_scaling=True,
        ),
        "linear_svm": ModelSpec(
            name="linear_svm",
            build=lambda: CalibratedClassifierCV(LinearSVC(class_weight="balanced"), cv=3),
            requires_scaling=True,
        ),
        "rbf_svm": ModelSpec(
            name="rbf_svm",
            build=lambda: SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=42),
            requires_scaling=True,
        ),
        "random_forest": ModelSpec(
            name="random_forest",
            build=lambda: RandomForestClassifier(
                n_estimators=300,
                random_state=42,
                class_weight="balanced",
                min_samples_leaf=2,
                n_jobs=-1,
            ),
            requires_scaling=False,
        ),
        "random_forest_classifier": ModelSpec(
            name="random_forest_classifier",
            build=lambda: RandomForestClassifier(
                n_estimators=300,
                random_state=42,
                class_weight="balanced",
                min_samples_leaf=2,
                n_jobs=-1,
            ),
            requires_scaling=False,
        ),
        "hist_gradient_boosting": ModelSpec(
            name="hist_gradient_boosting",
            build=lambda: HistGradientBoostingClassifier(random_state=42),
            requires_scaling=False,
        ),
        "stress_regressor_random_forest": ModelSpec(
            name="stress_regressor_random_forest",
            build=lambda: RandomForestRegressor(n_estimators=300, random_state=42, min_samples_leaf=2, n_jobs=-1),
            requires_scaling=False,
            prediction_mode="regressor",
        ),
        "stress_regressor_hist_gradient_boosting": ModelSpec(
            name="stress_regressor_hist_gradient_boosting",
            build=lambda: HistGradientBoostingRegressor(random_state=42),
            requires_scaling=False,
            prediction_mode="regressor",
        ),
        "svr_rbf": ModelSpec(
            name="svr_rbf",
            build=lambda: SVR(kernel="rbf", C=3.0, epsilon=0.05),
            requires_scaling=True,
            prediction_mode="regressor",
        ),
    }
    if name not in specs:
        raise KeyError(f"Unknown model spec: {name}")
    return specs[name]
