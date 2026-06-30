from __future__ import annotations

from kq_tool.utils.retry import is_nonempty_result, retry_call


class EmptyLike:
    empty = True


class NonEmptyLike:
    empty = False


def test_is_nonempty_result_handles_yfinance_like_empty_values() -> None:
    assert is_nonempty_result(None) is False
    assert is_nonempty_result(EmptyLike()) is False
    assert is_nonempty_result(NonEmptyLike()) is True
    assert is_nonempty_result({"ok": True}) is True


def test_retry_call_retries_exceptions_and_uses_backoff() -> None:
    calls = {"count": 0}
    sleeps = []

    def flaky():
        calls["count"] += 1
        if calls["count"] < 3:
            raise RuntimeError("temporary")
        return "ok"

    result = retry_call(
        flaky,
        tries=3,
        base_delay=2.0,
        jitter=0.5,
        sleep_fn=sleeps.append,
        random_fn=lambda _lo, _hi: 0.25,
    )

    assert result == "ok"
    assert calls["count"] == 3
    assert sleeps == [2.25, 4.25]


def test_retry_call_returns_none_after_empty_results() -> None:
    sleeps = []

    result = retry_call(
        lambda: EmptyLike(),
        tries=2,
        base_delay=1.0,
        jitter=0.0,
        sleep_fn=sleeps.append,
        random_fn=lambda _lo, _hi: 0.0,
    )

    assert result is None
    assert sleeps == [1.0]


def test_server_reuses_retry_helper() -> None:
    import server

    assert server._kq_retry_call is retry_call
