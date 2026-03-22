"""Shared artifact dataset constants and helpers."""

from __future__ import annotations

from typing import Iterable

ARTIFACT_CLASS_NAMES = ("clean", "eyem", "chew", "shiv", "elpp", "musc")
ARTIFACT_LABEL_TO_ID = {label: index for index, label in enumerate(ARTIFACT_CLASS_NAMES)}
ARTIFACT_ID_TO_LABEL = {index: label for label, index in ARTIFACT_LABEL_TO_ID.items()}
ARTIFACT_BINARY_NAMES = ("clean", "artifact")

_ARTIFACT_ALIASES = {
    "clean": "clean",
    "artifact_free": "clean",
    "none": "clean",
    "eyem": "eyem",
    "eye": "eyem",
    "eye_movement": "eyem",
    "eog": "eyem",
    "blink": "eyem",
    "chew": "chew",
    "shiv": "shiv",
    "shiver": "shiv",
    "elpp": "elpp",
    "electrode": "elpp",
    "electrode_pop": "elpp",
    "electrode_artifact": "elpp",
    "musc": "musc",
    "muscle": "musc",
    "emg": "musc",
}


def canonical_artifact_label(label: str) -> str:
    """Normalize an artifact label into the canonical label space."""
    normalized = label.strip().lower().replace("-", "_").replace(" ", "_")
    if normalized not in _ARTIFACT_ALIASES:
        raise ValueError(
            f"Unsupported artifact label '{label}'. Expected one of {sorted(set(_ARTIFACT_ALIASES.values()))}."
        )
    return _ARTIFACT_ALIASES[normalized]


def artifact_label_id(label: str) -> int:
    """Return the class ID for a canonical or aliased artifact label."""
    return ARTIFACT_LABEL_TO_ID[canonical_artifact_label(label)]


def artifact_binary_target(labels: Iterable[int]) -> list[int]:
    """Map multiclass artifact labels to clean-vs-artifact labels."""
    return [0 if int(label) == ARTIFACT_LABEL_TO_ID["clean"] else 1 for label in labels]
