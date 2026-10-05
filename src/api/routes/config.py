"""Config API: the Settings page reads and writes user settings.

The effective config a job runs with is config/config.yaml -> hardware
profile -> user settings (``data/settings.yaml``, written here; the file is
bind-mounted, untracked and may hold an API key) -> job profile -> job
overrides. ``config/config.yaml`` is never written: it is tracked by git, and
rewriting it dropped its comments and blocked ``git pull``.
"""

from pathlib import Path

import structlog
from fastapi import APIRouter, Depends, HTTPException
from pydantic import ValidationError

from src.api.routes.auth import get_current_user
from src.api.schemas.config import (
    ConfigResetRequest,
    ConfigResponse,
    ConfigUpdate,
    ProfileListResponse,
    ValidateRequest,
    ValidateResponse,
)
from src.api.services.auth import LocalUser
from src.config import Config, _deep_merge
from src.pipeline import run_pipeline
from src.user_settings import (
    ALLOWED_OVERRIDE_SECTIONS,
    apply_user_settings,
    load_user_settings,
    remove_user_keys,
    save_user_settings,
)

router = APIRouter()
logger = structlog.get_logger("config")

CONFIG_PATH = Path("config/config.yaml")
PROFILES_DIR = Path("config/profiles")

SECRET_KEYS = {"api_key", "secret", "password", "token"}
SECRET_PLACEHOLDER = "***"

# Marks "no default value at this path" (None itself is a valid default).
_NO_DEFAULT = object()


def _redact_secrets(data):
    """Recursively redact secret values before returning config to clients.

    Only an actually-set (truthy) secret is replaced with "***" — a null/
    empty one is returned as-is. That lets the frontend tell "a key is
    configured, value withheld" apart from "no key configured" from this
    response alone, without ever seeing the real value.
    """
    if isinstance(data, dict):
        result = {}
        for k, v in data.items():
            if k.lower() in SECRET_KEYS:
                result[k] = SECRET_PLACEHOLDER if v else v
            else:
                result[k] = _redact_secrets(v)
        return result
    if isinstance(data, list):
        return [_redact_secrets(v) for v in data]
    return data


def _strip_unchanged_secrets(updates):
    """Drop secret fields whose incoming value is the redaction placeholder.

    A client that fetched config via GET only ever sees "***" for a set
    secret and typically sends that same value straight back on PUT for
    fields it didn't touch. Without this, deep_merge would overwrite the
    real stored secret with the literal string "***". To actually clear a
    secret, a client must send "" or null, not "***".
    """
    if isinstance(updates, dict):
        cleaned = {}
        for k, v in updates.items():
            if k.lower() in SECRET_KEYS and v == SECRET_PLACEHOLDER:
                continue
            cleaned[k] = _strip_unchanged_secrets(v) if isinstance(v, dict) else v
        return cleaned
    return updates


def _same_value(value, default) -> bool:
    """Type-aware equality (in Python ``True == 1`` and ``1 == 1.0``)."""
    if isinstance(value, bool) or isinstance(default, bool):
        return isinstance(value, bool) == isinstance(default, bool) and value == default
    return value == default


def _drop_default_values(overrides, defaults) -> dict:
    """Drop leaves whose value equals the default, so the file stays minimal.

    Sections left without keys are dropped too. Paths the config does not
    know are kept as they are; pydantic decides about them at validation.
    """
    result = {}
    for key, value in overrides.items():
        default = defaults.get(key, _NO_DEFAULT) if isinstance(defaults, dict) else _NO_DEFAULT
        if isinstance(value, dict) and isinstance(default, dict):
            nested = _drop_default_values(value, default)
            if nested:
                result[key] = nested
        elif default is not _NO_DEFAULT and _same_value(value, default):
            continue
        else:
            result[key] = value
    return result


def _deep_copy(data):
    """A copy deep enough for config dicts (nested dicts and lists)."""
    if isinstance(data, dict):
        return {k: _deep_copy(v) for k, v in data.items()}
    if isinstance(data, list):
        return [_deep_copy(v) for v in data]
    return data


def _defaults() -> Config:
    """The effective config without user settings (what "changed" is against)."""
    if not CONFIG_PATH.exists():
        raise HTTPException(status_code=404, detail="Config file not found")
    return Config.load(str(CONFIG_PATH)).with_hardware_profile()


def _config_response() -> ConfigResponse:
    """The body of GET /config/ (and of PUT and POST /config/reset)."""
    defaults = _defaults()
    config, error = apply_user_settings(defaults)
    overrides, load_error = load_user_settings()
    if load_error:
        overrides = {}
    return ConfigResponse(
        config=_redact_secrets(config.model_dump()),
        defaults=_redact_secrets(defaults.model_dump()),
        overrides=_redact_secrets(overrides),
        error=error or load_error,
    )


@router.get("/", response_model=ConfigResponse)
async def get_config(user: LocalUser = Depends(get_current_user)):
    return _config_response()


@router.put("/", response_model=ConfigResponse)
async def update_config(
    update: ConfigUpdate,
    user: LocalUser = Depends(get_current_user),
):
    updates = _strip_unchanged_secrets(update.updates)
    blocked = sorted(set(updates) - ALLOWED_OVERRIDE_SECTIONS)
    if blocked:
        raise HTTPException(status_code=400, detail=f"Disallowed config override keys: {blocked}")

    defaults = _defaults().model_dump()
    current, error = load_user_settings()
    if error:
        # The file is broken, not secret: start over from the base config
        # (saving replaces it), but say what was ignored.
        logger.warning("user_settings_replaced", error=error)
        current = {}

    merged = current
    _deep_merge(merged, updates)
    overrides = _drop_default_values(merged, defaults)

    try:
        # _deep_merge works in place (returns None): copy first, merge on top.
        effective = _deep_copy(defaults)
        _deep_merge(effective, overrides)
        Config(**effective)
    except ValidationError as e:
        reason = "; ".join(
            f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in e.errors()
        )
        raise HTTPException(status_code=400, detail=f"Invalid config: {reason}")

    try:
        save_user_settings(overrides)
    except OSError as e:
        logger.error("user_settings_write_failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Cannot write the settings file: {e}")
    return _config_response()


@router.post("/reset", response_model=ConfigResponse)
async def reset_config(
    request: ConfigResetRequest,
    user: LocalUser = Depends(get_current_user),
):
    """Remove dotted keys from the user settings; an empty list removes all."""
    _defaults()  # 404 before touching anything if the base config is gone
    try:
        remove_user_keys(request.keys)
    except OSError as e:
        logger.error("user_settings_write_failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Cannot write the settings file: {e}")
    return _config_response()


@router.get("/profiles", response_model=ProfileListResponse)
async def list_profiles(user: LocalUser = Depends(get_current_user)):
    if not PROFILES_DIR.exists():
        return ProfileListResponse(profiles=[])

    profiles = [f.stem for f in PROFILES_DIR.glob("*.yaml")]
    return ProfileListResponse(profiles=profiles)


@router.post("/validate", response_model=ValidateResponse)
async def validate_config(
    request: ValidateRequest,
    user: LocalUser = Depends(get_current_user),
):
    try:
        config = Config(**request.config)

        run_pipeline(
            config=config,
            input_source=request.input_source,
            dry_run=True,
            resume=False,
            force=False,
        )

        return ValidateResponse(valid=True)
    except Exception as e:
        return ValidateResponse(valid=False, errors=[str(e)])
