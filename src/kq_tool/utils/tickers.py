"""Ticker normalization utilities."""

from __future__ import annotations


def ticker_to_code(ticker: str) -> str:
    """Convert Yahoo-style Korean tickers to six-digit local codes.

    Args:
        ticker: Examples include ``"005930.KS"``, ``"A005930"``, or ``"5930"``.

    Returns:
        A six-digit Korean stock/ETF code, e.g. ``"005930"``.
    """

    if ticker is None:
        raise ValueError("ticker must not be None")
    value = str(ticker).strip().upper()
    if not value:
        raise ValueError("ticker must not be empty")
    value = value.split(".")[0]
    value = value.removeprefix("A")
    return value.zfill(6)


# Temporary compatibility alias while legacy server.py is migrated.
_ticker_to_code = ticker_to_code
