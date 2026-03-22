"""Tests for STEW-style stress loading metadata."""

from __future__ import annotations

from src.config import Settings
from src.datasets.stress_local_loader import StressLocalLoader


def test_stress_loader_exposes_rating_target_score(synthetic_roots: dict[str, object]) -> None:
    settings = Settings(stress_data_root=synthetic_roots["stress_root"])
    loader = StressLocalLoader(settings=settings)
    bundle = loader.load_raw()
    assert "target_score" in bundle.metadata.columns
    assert bundle.metadata["target_score"].between(0.0, 1.0).all()
