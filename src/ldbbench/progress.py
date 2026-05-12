"""Small progress-reporting helpers for long-running CLI commands."""

from __future__ import annotations

import time
from collections.abc import Callable

ProgressCallback = Callable[[str], None]


class ProgressTicker:
    def __init__(
        self,
        progress: ProgressCallback | None,
        *,
        interval_seconds: float = 5.0,
    ) -> None:
        self._progress = progress
        self._interval_seconds = interval_seconds
        self._next_report_at = 0.0

    def emit(self, message: str) -> None:
        if self._progress is not None:
            self._progress(message)
        self._next_report_at = time.perf_counter() + self._interval_seconds

    def maybe(self, message: str) -> None:
        if self._progress is None:
            return
        now = time.perf_counter()
        if now < self._next_report_at:
            return
        self._progress(message)
        self._next_report_at = now + self._interval_seconds
