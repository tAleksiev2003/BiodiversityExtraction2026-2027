"""Tests for src/settings.py."""
from pathlib import Path

from settings import deep_merge, load_settings

FIXTURES = Path(__file__).parent / "fixtures"


def test_deep_merge_keeps_untouched_keys():
    merged = deep_merge({"a": {"x": 1, "y": 2}, "b": 3}, {"a": {"x": 10}})
    assert merged == {"a": {"x": 10, "y": 2}, "b": 3}


def test_experiment_overrides_one_value_only():
    base = load_settings()
    experiment = load_settings(FIXTURES / "experiment_override.yaml")
    assert experiment["chunking"]["max_chunk_tokens"] == 1234
    assert experiment["chunking"]["chunk_overlap_tokens"] == base["chunking"]["chunk_overlap_tokens"]


def test_paths_are_absolute():
    settings = load_settings()
    assert all(path.is_absolute() for path in settings["paths"].values())
