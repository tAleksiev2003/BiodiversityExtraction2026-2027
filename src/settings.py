"""Load all settings from config/settings.yaml (plus an optional experiment file).

Every threshold, path and model name lives in the YAML file, so the code never contains
magic numbers (SPEC.md, section 10). An experiment file in config/experiments/ only lists
what changes; it is merged on top of the base settings.
"""
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
BASE_SETTINGS_PATH = REPO_ROOT / "config" / "settings.yaml"


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of `base` where nested keys from `override` replace those in `base`.

    A plain dict.update would replace a whole section (e.g. all of `chunking`) when an
    experiment changes only one value in it.
    """
    merged = dict(base)
    for key, override_value in override.items():
        base_value = merged.get(key)
        if isinstance(base_value, dict) and isinstance(override_value, dict):
            merged[key] = deep_merge(base_value, override_value)
        else:
            merged[key] = override_value
    return merged


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        content = yaml.safe_load(file)
    return content or {}


def _resolve_paths(paths: dict[str, str], root: Path) -> dict[str, Path]:
    """Make all paths absolute, so scripts work from any working directory."""
    return {name: (root / value).resolve() for name, value in paths.items()}


def load_settings(
    experiment_path: Path | None = None,
    base_path: Path = BASE_SETTINGS_PATH,
    root: Path = REPO_ROOT,
) -> dict[str, Any]:
    """Load the base settings, merge an optional experiment file and resolve paths."""
    settings = _read_yaml(base_path)
    if experiment_path is not None:
        settings = deep_merge(settings, _read_yaml(experiment_path))
    settings["paths"] = _resolve_paths(settings.get("paths", {}), root)
    return settings
