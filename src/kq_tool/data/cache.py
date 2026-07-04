"""Small TTL cache helpers used by the legacy server and new modules."""

from __future__ import annotations

import time
from collections.abc import Callable, MutableMapping
from typing import TypeVar

T = TypeVar("T")
CacheStore = MutableMapping[str, tuple[float, object]]


def cached(
    store: CacheStore,
    key: str,
    ttl: float,
    fn: Callable[..., T],
    *args: object,
    now_fn: Callable[[], float] = time.time,
) -> T:
    """Return cached result while the entry is younger than ``ttl`` seconds."""

    now = now_fn()
    if key in store and now - store[key][0] < ttl:
        return store[key][1]  # type: ignore[return-value]
    result = fn(*args)
    store[key] = (now, result)
    return result


def clear_expired(store: CacheStore, ttl: float, now_fn: Callable[[], float] = time.time) -> int:
    """Remove expired entries from a cache store and return the number removed."""

    now = now_fn()
    expired = [key for key, (created_at, _) in store.items() if now - created_at >= ttl]
    for key in expired:
        del store[key]
    return len(expired)
