"""I/O helpers for artifacts and data files."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import pandas as pd


def ensure_dir(path: Path) -> Path:
    """Create a directory if it does not exist."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json_data(data: Any, path: Path) -> None:
    """Serialize JSON-compatible data with stable formatting."""
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)


def save_json(data: dict[str, Any], path: Path) -> None:
    """Serialize a JSON object with stable formatting."""
    save_json_data(data, path)


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON file."""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_pickle(obj: Any, path: Path) -> None:
    """Serialize an object with pickle."""
    ensure_dir(path.parent)
    with path.open("wb") as handle:
        pickle.dump(obj, handle)


def load_pickle(path: Path) -> Any:
    """Load a pickle file."""
    with path.open("rb") as handle:
        return pickle.load(handle)


def save_dataframe(frame: pd.DataFrame, path: Path) -> None:
    """Write a DataFrame to CSV."""
    ensure_dir(path.parent)
    frame.to_csv(path, index=False)
