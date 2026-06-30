from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kq_tool.analyzer.indicators import atr, bollinger_bands, close_series, macd, rsi


@pytest.fixture
def close() -> pd.Series:
    index = pd.date_range("2024-01-01", periods=80, freq="B")
    values = np.linspace(100, 130, len(index)) + np.sin(np.arange(len(index))) * 2
    return pd.Series(values, index=index, name="Close")


def test_close_series_accepts_series_and_dataframe(close: pd.Series) -> None:
    assert close_series(close).equals(close)
    assert close_series(pd.DataFrame({"Close": close})).equals(close)


def test_rsi_returns_bounded_values_after_warmup(close: pd.Series) -> None:
    values = rsi(close).dropna()

    assert not values.empty
    assert values.between(0, 100).all()


def test_macd_preserves_index(close: pd.Series) -> None:
    macd_line, signal_line = macd(close)

    assert macd_line.index.equals(close.index)
    assert signal_line.index.equals(close.index)
    assert len(macd_line) == len(signal_line) == len(close)


def test_bollinger_bands_are_ordered_after_warmup(close: pd.Series) -> None:
    upper, middle, lower = bollinger_bands(close)
    valid = pd.DataFrame({"upper": upper, "middle": middle, "lower": lower}).dropna()

    assert not valid.empty
    assert (valid["upper"] >= valid["middle"]).all()
    assert (valid["middle"] >= valid["lower"]).all()


def test_atr_requires_ohlc_columns(close: pd.Series) -> None:
    with pytest.raises(ValueError):
        atr(pd.DataFrame({"Close": close}))


def test_atr_returns_non_negative_series(close: pd.Series) -> None:
    ohlc = pd.DataFrame(
        {
            "High": close + 1.5,
            "Low": close - 1.0,
            "Close": close,
        }
    )

    values = atr(ohlc).dropna()
    assert not values.empty
    assert (values >= 0).all()


def test_server_reuses_indicator_helpers(close: pd.Series) -> None:
    import server

    assert server._kq_close_series is close_series
    assert server._kq_rsi is rsi
    assert server._kq_macd is macd
    assert server._kq_bb is bollinger_bands
    assert server._kq_atr is atr
    assert server._c(pd.DataFrame({"Close": close})).equals(close)
    assert server._rsi(close).equals(rsi(close))
