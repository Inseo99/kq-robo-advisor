from __future__ import annotations

import pytest

from kq_tool.utils.tickers import _ticker_to_code, ticker_to_code


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("005930.KS", "005930"),
        ("005930.KQ", "005930"),
        ("A005930", "005930"),
        ("5930", "005930"),
        ("  069500.ks ", "069500"),
    ],
)
def test_ticker_to_code_normalizes_korean_tickers(raw: str, expected: str) -> None:
    assert ticker_to_code(raw) == expected


def test_legacy_alias_matches_public_function() -> None:
    assert _ticker_to_code("005930.KS") == ticker_to_code("005930.KS")


@pytest.mark.parametrize("raw", ["", "   ", None])
def test_ticker_to_code_rejects_empty_values(raw: str | None) -> None:
    with pytest.raises(ValueError):
        ticker_to_code(raw)  # type: ignore[arg-type]
