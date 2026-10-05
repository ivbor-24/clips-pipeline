"""Build the pipeline config of a web job from its stored overrides.

Separate from src.api.services.job so the job worker (src/worker.py) can use it
without loading the API settings and database engine.
"""

import json
from pathlib import Path
from typing import Optional

from src.config import Config, _deep_merge
from src.user_settings import ALLOWED_OVERRIDE_SECTIONS, load_app_config


def build_job_config(
    config_overrides: Optional[str],
    work_dir: str,
    base_config_path: str = "config/config.yaml",
) -> Config:
    """Build the pipeline config for a job.

    Args:
        config_overrides: JSON from the job record. An optional "profile" key
            is applied first (same semantics as CLI --profile); the remaining
            section overrides are merged on top of it.
        work_dir: Job working directory.
        base_config_path: Base YAML config.

    Returns:
        Validated Config with work_dir set to the job directory.

    Raises:
        ValueError: On a disallowed override key or an invalid profile name.
    """
    # config.yaml -> hardware profile -> user settings (data/settings.yaml,
    # the Settings page); the job profile and overrides go on top of it.
    config = load_app_config(base_config_path)
    if not config_overrides:
        config.work_dir = Path(work_dir)
        return config

    overrides = json.loads(config_overrides)
    profile_name = overrides.pop("profile", None)
    blocked = sorted(set(overrides.keys()) - ALLOWED_OVERRIDE_SECTIONS)
    if blocked:
        raise ValueError(f"Disallowed config override keys: {blocked}")

    if profile_name:
        config = config.load_profile(str(profile_name))

    merged = config.model_dump()
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            _deep_merge(merged[key], value)
        else:
            merged[key] = value
    merged["work_dir"] = str(work_dir)
    return Config(**merged)
