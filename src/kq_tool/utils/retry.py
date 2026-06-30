"""Retry helpers for transient external-service failures."""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")


def is_nonempty_result(value: object) -> bool:
    """Return True when a result is usable for yfinance-style responses."""

    if value is None:
        return False
    if hasattr(value, "empty") and bool(getattr(value, "empty")):
        return False
    return True


def retry_call(
    fn: Callable[[], T],
    *,
    tries: int = 3,
    base_delay: float = 1.0,
    jitter: float = 0.4,
    sleep_fn: Callable[[float], None] = time.sleep,
    random_fn: Callable[[float, float], float] = random.uniform,
    is_success: Callable[[object], bool] = is_nonempty_result,
) -> T | None:
    """Call ``fn`` with exponential backoff and jitter.

    This is intentionally small and dependency-free so it can be used by the
    legacy server while remaining easy to unit test.
    """

    for attempt in range(max(0, int(tries))):
        try:
            result = fn()
            if is_success(result):
                return result
        except Exception:
            pass

        if attempt < tries - 1:
            sleep_fn(base_delay * (2**attempt) + random_fn(0, jitter))

    return None


_retry_call = retry_call
