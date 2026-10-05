"""User settings from the web UI: data/settings.yaml.

The Settings page saves here, not into config/config.yaml (which is tracked
by git and bind-mounted read-only into the containers; rewriting it dropped
its comments and blocked ``git pull``). Only the keys the user changed are
stored, as a nested mapping, e.g. ``{scoring: {max_clips_per_video: 8}}``.

The path comes from the ``PIPELINE_USER_SETTINGS`` environment variable
(default ``data/settings.yaml`` relative to the working directory: ``/app``
in Docker, the checkout natively); ``data/`` is already bind-mounted,
writable and untracked.

Layering (see ``load_app_config``): config/config.yaml -> hardware profile
(``with_hardware_profile``) -> user settings -> job profile -> job overrides.
The CLI (``src/cli.py``) deliberately stays out: it keeps reading only the
config file it is given, so a run stays reproducible (see docs/CONFIG.md).
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import structlog
import yaml
from pydantic import ValidationError

from src.config import Config, _deep_merge

logger = structlog.get_logger("user_settings")

# Top-level config sections the user (Settings page) or a job may override;
# dangerous keys (work_dir, gpu providers, profiles_dir) stay blocked. Shared
# by src/user_settings.py and src/api/job_config.py; keep one list for both.
ALLOWED_OVERRIDE_SECTIONS = {
    "input",
    "preprocessing",
    "transcription",
    "scoring",
    "cropping",
    "rendering",
    "chapters",
    "broll",
    "cleanup",
    "output",
    "logging",
}

DEFAULT_SETTINGS_PATH = Path("data/settings.yaml")
# The file may hold an API key (scoring.llm.api_key).
SETTINGS_MODE = 0o600


def user_settings_path() -> Path:
    """Where the user settings live (PIPELINE_USER_SETTINGS, default data/settings.yaml)."""
    return Path(os.environ.get("PIPELINE_USER_SETTINGS") or DEFAULT_SETTINGS_PATH)


def load_user_settings() -> Tuple[Dict[str, Any], Optional[str]]:
    """Read the user settings file: ``(overrides, error)``.

    A missing or empty file means no user settings: ``({}, None)``. An
    unreadable or invalid file (bad YAML, not a mapping, disallowed sections)
    must not stop the API or the worker: the caller gets ``({}, reason)`` and
    shows the reason through ``GET /config/`` (the ``error`` field).
    """
    path = user_settings_path()
    if not path.exists():
        return {}, None
    try:
        with open(path) as f:
            data = yaml.safe_load(f)
    except (OSError, yaml.YAMLError) as e:
        return {}, f"Cannot read {path}: {e}"
    if data is None:  # an empty file
        return {}, None
    if not isinstance(data, dict):
        return {}, f"{path} must hold a mapping of config sections, not {type(data).__name__}"
    blocked = sorted(set(data) - ALLOWED_OVERRIDE_SECTIONS)
    if blocked:
        return {}, f"Disallowed sections in {path}: {blocked}"
    non_mapping = sorted(k for k, v in data.items() if not isinstance(v, dict))
    if non_mapping:
        return {}, f"Section values must be mappings in {path}: {non_mapping}"
    return data, None


def save_user_settings(overrides: Dict[str, Any]) -> None:
    """Write the user settings file atomically (temp file + rename), mode 0600.

    Raises:
        OSError: If the file cannot be written; the caller turns it into an
            error the page shows.
    """
    path = user_settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    # Created 0600 from the start: the file may hold an API key.
    fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, SETTINGS_MODE)
    with os.fdopen(fd, "w") as f:
        if overrides:
            yaml.safe_dump(overrides, f, default_flow_style=False, sort_keys=True)
    os.chmod(temp, SETTINGS_MODE)  # an older temp file kept its mode
    os.replace(temp, path)


def apply_user_settings(config: Config) -> Tuple[Config, Optional[str]]:
    """Merge the user settings into ``config``: ``(config, error)``.

    On error the base ``config`` is returned unchanged, so a broken settings
    file never stops the API or the worker; the reason is also returned for
    ``GET /config/`` to show.
    """
    overrides, error = load_user_settings()
    if error or not overrides:
        return config, error
    merged = config.model_dump()
    _deep_merge(merged, overrides)
    try:
        return Config(**merged), None
    except ValidationError as e:
        reason = "; ".join(
            f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in e.errors()
        )
        return config, f"Invalid user settings: {reason}"


def load_app_config(base_path: str = "config/config.yaml") -> Config:
    """The effective base config for the API and the worker.

    config/config.yaml -> hardware profile -> user settings. A broken user
    settings file is logged and ignored (the caller cannot act on it; the
    Settings page shows the reason via ``GET /config/``).

    Raises:
        FileNotFoundError: If ``base_path`` does not exist (fail fast, as
            with ``Config.load``).
    """
    config = Config.load(base_path).with_hardware_profile()
    config, error = apply_user_settings(config)
    if error:
        logger.warning("user_settings_ignored", error=error)
    return config


def remove_user_keys(dotted_keys: List[str]) -> Dict[str, Any]:
    """Remove dotted keys (``scoring.max_clips_per_video``) from the user settings.

    An empty list removes everything (all settings go back to their
    defaults). Returns the saved settings. Sections unknown to the config
    or keys that are not set are ignored: removing them changes nothing.

    Raises:
        OSError: If the file cannot be written.
    """
    overrides, _error = load_user_settings()
    if dotted_keys:
        for key in dotted_keys:
            _remove_dotted(overrides, key)
        overrides = _prune_empty(overrides)
    else:
        overrides = {}  # an empty list resets everything
    save_user_settings(overrides)
    return overrides


def _remove_dotted(data: Dict[str, Any], dotted: str) -> None:
    parts = dotted.split(".")
    node = data
    for part in parts[:-1]:
        if not isinstance(node.get(part), dict):
            return
        node = node[part]
    node.pop(parts[-1], None)


def _prune_empty(data: Dict[str, Any]) -> Dict[str, Any]:
    """Drop sections left without keys, so the file stays minimal."""
    return {
        key: (_prune_empty(value) if isinstance(value, dict) else value)
        for key, value in data.items()
        if not isinstance(value, dict) or _prune_empty(value)
    }
