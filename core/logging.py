"""
Structured JSON logger for VigilAI.
Provides a consistent, production-grade logging interface across all modules.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from datetime import datetime, timezone
from typing import Any

from core.config import get_settings


class _JSONFormatter(logging.Formatter):
    """Formats log records as newline-delimited JSON (NDJSON)."""

    def format(self, record: logging.LogRecord) -> str:  # noqa: A003
        payload: dict[str, Any] = {
            "ts": datetime.now(tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "func": record.funcName,
            "line": record.lineno,
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        if hasattr(record, "extra"):
            payload.update(record.extra)  # type: ignore[arg-type]
        return json.dumps(payload, ensure_ascii=False)


class _TextFormatter(logging.Formatter):
    _COLORS = {
        "DEBUG": "\033[36m",
        "INFO": "\033[32m",
        "WARNING": "\033[33m",
        "ERROR": "\033[31m",
        "CRITICAL": "\033[35m",
    }
    _RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:  # noqa: A003
        color = self._COLORS.get(record.levelname, "")
        ts = datetime.now(tz=timezone.utc).strftime("%H:%M:%S")
        return (
            f"{color}[{ts}] {record.levelname:<8}{self._RESET} "
            f"{record.name} — {record.getMessage()}"
        )


def get_logger(name: str) -> logging.Logger:
    """
    Returns a configured logger.  Call once per module:

        log = get_logger(__name__)
    """
    settings = get_settings()
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger  # already configured

    logger.setLevel(settings.log_level)

    handler = logging.StreamHandler(sys.stdout)
    if settings.log_format == "json":
        handler.setFormatter(_JSONFormatter())
    else:
        handler.setFormatter(_TextFormatter())

    logger.addHandler(handler)
    logger.propagate = False
    return logger


class Timer:
    """Context manager to measure and log elapsed time."""

    def __init__(self, label: str, logger: logging.Logger | None = None) -> None:
        self._label = label
        self._log = logger
        self._start: float = 0.0
        self.elapsed: float = 0.0

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_: Any) -> None:
        self.elapsed = time.perf_counter() - self._start
        if self._log:
            self._log.info("%s completed in %.3fs", self._label, self.elapsed)
