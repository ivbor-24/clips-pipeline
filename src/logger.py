import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, List, Optional

import structlog

_SHARED_PROCESSORS: List[Any] = [
    structlog.contextvars.merge_contextvars,
    structlog.processors.add_log_level,
    structlog.processors.TimeStamper(fmt="iso"),
]


def _make_formatter(json_format: bool) -> structlog.stdlib.ProcessorFormatter:
    renderer = (
        structlog.processors.JSONRenderer()
        if json_format
        else structlog.dev.ConsoleRenderer(colors=True)
    )
    return structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=_SHARED_PROCESSORS,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )


def setup_logging(
    log_file: Optional[str] = "artifacts/logs/pipeline.log",
    level: str = "INFO",
    json_format: bool = True,
) -> None:
    """Configure structured logging to console and, optionally, a file.

    Uses structlog's stdlib integration so that both structlog and standard
    `logging` records are routed through the same handlers (console + file).

    Args:
        log_file: Log file path; None logs to the console only.
        level: Minimum level name.
        json_format: JSON lines instead of the colored console renderer.
    """
    level_num = getattr(logging, level.upper(), logging.INFO)
    formatter = _make_formatter(json_format)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(level_num)
    # Avoid duplicate handlers on re-configuration.
    root_logger.handlers.clear()
    root_logger.addHandler(console_handler)

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    structlog.configure(
        processors=[
            *_SHARED_PROCESSORS,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


def get_logger(name: str = "pipeline") -> structlog.BoundLogger:
    """Get a logger instance with the given name."""
    return structlog.get_logger(name)


class _JobRecordFilter(logging.Filter):
    """Pass only structlog records bound to one job."""

    def __init__(self, job_id: Any) -> None:
        super().__init__()
        self.job_id = job_id

    def filter(self, record: logging.LogRecord) -> bool:
        event = record.msg
        return isinstance(event, dict) and event.get("job_id") == self.job_id


@contextmanager
def job_log_file(log_file: Path, job_id: Any) -> Iterator[None]:
    """Mirror one job's structured logs into its own JSON-lines file.

    Binds ``job_id`` into structlog contextvars; asyncio tasks and
    ``asyncio.to_thread`` copy the context, so records from the pipeline
    thread carry it and concurrent jobs in one process never mix. Requires
    ``setup_logging()`` (stdlib-backed structlog).

    Args:
        log_file: Destination file (parent directories are created).
        job_id: Job identifier bound to every record of the job.
    """
    log_file.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_file, encoding="utf-8")
    handler.setFormatter(_make_formatter(json_format=True))
    handler.addFilter(_JobRecordFilter(job_id))
    root_logger = logging.getLogger()
    root_logger.addHandler(handler)
    tokens = structlog.contextvars.bind_contextvars(job_id=job_id)
    try:
        yield
    finally:
        structlog.contextvars.reset_contextvars(**tokens)
        root_logger.removeHandler(handler)
        handler.close()
