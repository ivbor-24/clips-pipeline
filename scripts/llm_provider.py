#!/usr/bin/env python3
"""
Which LLM chooses the clips: setup.sh's helper, run in the job worker container.

The LLM is required: either the local model
(scoring.llm.provider llama_cpp, Qwen3-14B) or an API provider. The choice
goes into the user settings (data/settings.yaml, the file the Settings page
writes), the API key into .env (setup.sh does that).

Usage:
    python scripts/llm_provider.py status
        LLM_PROVIDER=..., LLM_MODEL=..., LLM_READY=yes|no lines for setup.sh
    python scripts/llm_provider.py use-api --provider openai|anthropic --model NAME [--api-base URL]
    python scripts/llm_provider.py use-local
    python scripts/llm_provider.py check
        One short request to the API provider; exit code 1 with the reason if it fails.
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import structlog

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import LLMConfig  # noqa: E402
from src.user_settings import (  # noqa: E402
    load_app_config,
    load_user_settings,
    remove_user_keys,
    save_user_settings,
)

API_PROVIDERS = ("openai", "anthropic")
KEY_VARIABLES = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}
# User keys that pick the LLM; use-api / use-local rewrite all of them.
CHOICE_KEYS = [
    "scoring.llm.enabled",
    "scoring.llm.provider",
    "scoring.llm.model",
    "scoring.llm.api_base",
    "scoring.llm.api_key",
]
CHECK_PROMPT = 'Respond with JSON only, exactly: {"results": [{"ok": true}]}'


def has_api_key(llm: LLMConfig) -> bool:
    variable = KEY_VARIABLES.get(llm.provider)
    return bool(llm.api_key or (variable and os.environ.get(variable)))


def status(llm: LLMConfig) -> dict:
    """What the jobs would use now, and whether it can work without more setup."""
    if not llm.enabled:
        ready = False
    elif llm.provider in API_PROVIDERS:
        ready = has_api_key(llm)
    elif llm.provider == "llama_cpp":
        ready = bool(llm.model_path) and Path(llm.model_path).exists()
    else:
        ready = True  # qwen_local: a hand-made config, not setup.sh's business
    return {
        "LLM_ENABLED": "yes" if llm.enabled else "no",
        "LLM_PROVIDER": llm.provider,
        "LLM_MODEL": llm.model,
        "LLM_API_BASE": llm.api_base or "",
        "LLM_READY": "yes" if ready else "no",
    }


def use_api(provider: str, model: str, api_base: str = "") -> None:
    overrides = remove_user_keys(CHOICE_KEYS)
    llm = {"provider": provider, "model": model}
    if api_base:
        llm["api_base"] = api_base
    overrides.setdefault("scoring", {}).setdefault("llm", {}).update(llm)
    save_user_settings(overrides)


def check(llm: LLMConfig) -> str:
    """Send one short request through the adapter the jobs use; returns a summary.

    Raises:
        LLMAdapterError: The request failed or the answer is not the JSON asked for.
    """
    from src.llm_adapter import LLMAdapterError, create_llm_adapter

    if llm.provider not in API_PROVIDERS:
        raise LLMAdapterError(f"check works for API providers only, not {llm.provider}")
    adapter = create_llm_adapter(llm)
    started = time.monotonic()
    result = adapter.analyze_segments([], prompt=CHECK_PROMPT)
    if not result:
        raise LLMAdapterError("the model answered, but not with the JSON asked for")
    return f"{llm.provider} model {llm.model} answered in {time.monotonic() - started:.1f} s"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status")
    api = sub.add_parser("use-api")
    api.add_argument("--provider", choices=API_PROVIDERS, required=True)
    api.add_argument("--model", required=True)
    api.add_argument("--api-base", default="")
    sub.add_parser("use-local")
    sub.add_parser("check")
    args = parser.parse_args()
    # setup.sh shows this script's output: the result, not the adapter's logs.
    structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(logging.CRITICAL))

    if args.command in ("use-api", "use-local"):
        _settings, error = load_user_settings()
        if error:
            # Rewriting would drop the user's other settings.
            print(f"Cannot change the settings: {error}", file=sys.stderr)
            return 1
        if args.command == "use-api":
            use_api(args.provider, args.model, args.api_base)
        else:
            remove_user_keys(CHOICE_KEYS)
        return 0

    llm = load_app_config().scoring.llm
    if args.command == "status":
        for key, value in status(llm).items():
            print(f"{key}={value}")
        return 0
    try:
        print(check(llm))
    except Exception as e:
        print(str(e), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
