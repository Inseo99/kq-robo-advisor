from __future__ import annotations

from kq_tool.data.cache import cached, clear_expired


def test_cached_reuses_value_before_ttl() -> None:
    store = {}
    calls = {"n": 0}

    def compute() -> int:
        calls["n"] += 1
        return calls["n"]

    assert cached(store, "x", 10, compute, now_fn=lambda: 100.0) == 1
    assert cached(store, "x", 10, compute, now_fn=lambda: 105.0) == 1
    assert calls["n"] == 1


def test_cached_refreshes_value_after_ttl() -> None:
    store = {}
    calls = {"n": 0}

    def compute() -> int:
        calls["n"] += 1
        return calls["n"]

    assert cached(store, "x", 10, compute, now_fn=lambda: 100.0) == 1
    assert cached(store, "x", 10, compute, now_fn=lambda: 111.0) == 2


def test_clear_expired_removes_only_old_entries() -> None:
    store = {"old": (100.0, 1), "fresh": (109.0, 2)}

    removed = clear_expired(store, 10, now_fn=lambda: 111.0)

    assert removed == 1
    assert store == {"fresh": (109.0, 2)}
