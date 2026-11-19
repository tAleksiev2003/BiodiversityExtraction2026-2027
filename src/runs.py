"""Create one folder per run, so every result can be traced back to its settings and code.

Each run folder gets the merged settings and the git commit hash (SPEC.md, section 8).
The rest of the run logging (prompts, answers, energy) comes with the later milestones.
"""
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from settings import REPO_ROOT

TIMESTAMP_FORMAT = "%Y%m%d-%H%M%S"


def git_commit_hash() -> str:
    """Current commit, with a "-dirty" mark if there are uncommitted changes."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return f"{commit}-dirty" if status else commit


def _paths_as_strings(settings: dict[str, Any]) -> dict[str, Any]:
    """YAML cannot write Path objects; store them as plain strings."""
    copied = dict(settings)
    copied["paths"] = {name: str(path) for name, path in settings.get("paths", {}).items()}
    return copied


def create_run_dir(settings: dict[str, Any], experiment_name: str) -> Path:
    """Create runs/<timestamp>_<experiment>/ with config.yaml and commit.txt."""
    timestamp = datetime.now().strftime(TIMESTAMP_FORMAT)
    run_dir = settings["paths"]["runs"] / f"{timestamp}_{experiment_name}"
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(_paths_as_strings(settings), allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    (run_dir / "commit.txt").write_text(git_commit_hash() + "\n", encoding="utf-8")
    return run_dir


def load_run_settings(run_dir: Path) -> dict[str, Any]:
    """The settings an earlier run used (its config.yaml), with the paths as Path objects.

    A later step that builds on a run (src/run_verify.py) must rebuild that run's prompts
    exactly, so it uses these settings and not the current ones.
    """
    config_path = run_dir / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"{config_path} missing: is {run_dir} a run folder?")
    settings = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    settings["paths"] = {name: Path(value) for name, value in settings["paths"].items()}
    return settings
