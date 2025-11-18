from __future__ import annotations

import logging
from collections import deque
from typing import Deque, List

from .time_utils import now_tz


class InMemoryLogBuffer:
    def __init__(self, max_lines: int = 500) -> None:
        self._buffer: Deque[str] = deque(maxlen=max_lines)

    def append(self, line: str) -> None:
        self._buffer.append(line)

    def list(self, limit: int | None = None) -> List[str]:
        lines = list(self._buffer)
        return lines[-limit:] if limit else lines


log_buffer = InMemoryLogBuffer()


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("pytrader")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M")
    stream = logging.StreamHandler()
    stream.setFormatter(fmt)
    logger.addHandler(stream)
    return logger


logger = setup_logger()


def log_line(message: str) -> None:
    ts = now_tz().strftime("%H:%M")
    line = f"[{ts}] {message}"
    logger.info(message)
    log_buffer.append(line)


