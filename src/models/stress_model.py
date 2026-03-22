"""Stress model defaults and label maps."""

STRESS_CLASS_NAMES = ["natural", "low", "mid", "high"]
STRESS_CLASS_TO_SCORE = {0: 0.0, 1: 0.33, 2: 0.66, 3: 1.0}
DEFAULT_STRESS_CANDIDATES = ("svr_rbf", "stress_regressor_random_forest", "stress_regressor_hist_gradient_boosting")
