"""Pure technical indicator functions.

These functions intentionally have no data-loading or application state.
They are safe to unit test and reuse from both the legacy server and the
new modular package.
"""

from __future__ import annotations

import pandas as pd


def close_series(data: pd.DataFrame | pd.Series) -> pd.Series:
    """Return a 1-dimensional close-price series from a Series/DataFrame."""

    if isinstance(data, pd.DataFrame):
        close = data["Close"] if "Close" in data.columns else data.iloc[:, 0]
    else:
        close = data
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    return close.squeeze()


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Compute simple rolling RSI."""

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean().replace(0, 1e-9)
    return 100 - 100 / (1 + gain / loss)


def macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[pd.Series, pd.Series]:
    """Compute MACD line and signal line."""

    macd_line = close.ewm(span=fast, adjust=False).mean() - close.ewm(
        span=slow, adjust=False
    ).mean()
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line


def bollinger_bands(
    close: pd.Series,
    window: int = 20,
    num_std: float = 2.0,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Compute upper, middle, and lower Bollinger Bands."""

    middle = close.rolling(window).mean()
    std = close.rolling(window).std()
    return middle + num_std * std, middle, middle - num_std * std


def atr(ohlc: pd.DataFrame, window: int = 14) -> pd.Series:
    """Compute Average True Range from OHLC data."""

    required = {"High", "Low", "Close"}
    missing = required.difference(ohlc.columns)
    if missing:
        raise ValueError(f"missing OHLC columns: {sorted(missing)}")
    high = ohlc["High"].squeeze()
    low = ohlc["Low"].squeeze()
    prev_close = ohlc["Close"].squeeze().shift(1)
    true_range = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return true_range.rolling(window).mean()


# Temporary compatibility aliases while legacy server.py is migrated.
_c = close_series
_rsi = rsi
_macd = macd
_bb = bollinger_bands
_atr = atr
