"""
Notices — problems that do not stop a job but change its result

Example: the LLM did not load, so clips were picked by heuristics and are
likely worse. Such a fallback used to be a warning in ``job.log`` only; the
user reads the job page. A stage records a notice here, the API returns the
notices of a job, and the job page shows them.

Notices are grouped by stage. A stage clears its own notices when it runs
again (resume, retry), so a fallback that did not repeat stops being shown.

Inputs / Outputs:
- artifacts/notices.json: {"<stage>": [{"code", "message", ...details}]}
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Union

import structlog

logger = structlog.get_logger("notices")

NOTICES_FILE = Path("artifacts") / "notices.json"


def _path(work_dir: Union[str, Path]) -> Path:
    return Path(work_dir) / NOTICES_FILE


def _load(work_dir: Union[str, Path]) -> Dict[str, List[Dict[str, Any]]]:
    path = _path(work_dir)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("notices_unreadable", path=str(path), error=str(e))
        return {}
    return data if isinstance(data, dict) else {}


def _save(work_dir: Union[str, Path], data: Dict[str, List[Dict[str, Any]]]) -> None:
    path = _path(work_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def clear_notices(work_dir: Union[str, Path], stage: str) -> None:
    """Forget the notices of ``stage``; call when the stage starts."""
    data = _load(work_dir)
    if data.pop(stage, None) is not None:
        _save(work_dir, data)


def add_notice(
    work_dir: Union[str, Path], stage: str, code: str, message: str, **details: Any
) -> None:
    """Record a notice for the user and log it as a warning.

    Args:
        work_dir: The job's work directory.
        stage: Pipeline stage that hit the problem (``scoring``).
        code: Stable identifier (``llm_fallback_heuristics``).
        message: One sentence for the user: what happened and what it means.
        **details: Extra JSON-serializable facts (the error text).
    """
    logger.warning(code, stage=stage, message=message, **details)
    data = _load(work_dir)
    data.setdefault(stage, []).append({"code": code, "message": message, **details})
    _save(work_dir, data)


def read_notices(work_dir: Union[str, Path]) -> List[Dict[str, Any]]:
    """All notices of a job, each with its ``stage``."""
    return [
        {"stage": stage, **notice}
        for stage, notices in _load(work_dir).items()
        for notice in notices
    ]
