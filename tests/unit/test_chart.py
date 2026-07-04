from __future__ import annotations

import numpy as np
import pandas as pd

from kq_tool.analyzer.chart import build_stock_chart


def _ohlcv(rows: int = 600) -> pd.DataFrame:
    index = pd.date_range("2020-01-02", periods=rows, freq="B")
    close = pd.Series(np.linspace(50_000, 80_000, rows), index=index)
    return pd.DataFrame(
        {
            "Open": close - 100,
            "High": close + 500,
            "Low": close - 500,
            "Close": close,
            "Volume": np.arange(rows) + 1_000,
        },
        index=index,
    )


def test_build_stock_chart_preserves_full_requested_history() -> None:
    frame = _ohlcv(rows=600)
    close = frame["Close"]

    chart = build_stock_chart(
        frame,
        close,
        indicators={
            "ma20": close.rolling(20).mean(),
            "ma60": close.rolling(60).mean(),
            "ma200": close.rolling(200).mean(),
        },
        period="max",
    )

    assert chart["period"] == "max"
    assert chart["count"] == 600
    assert len(chart["dates"]) == 600
    assert len(chart["close"]) == 600
    assert chart["dates"][0] == "2020-01-02"
    assert chart["dates"][-1] == frame.index[-1].strftime("%Y-%m-%d")


def test_build_stock_chart_derives_macd_histogram_when_missing() -> None:
    frame = _ohlcv(rows=30)
    close = frame["Close"]
    macd = pd.Series(np.linspace(-1.0, 1.0, len(close)), index=close.index)
    signal = pd.Series(0.25, index=close.index)

    chart = build_stock_chart(
        frame,
        close,
        indicators={"macd": macd, "signal": signal},
    )

    assert chart["hist"][0] == -1.25
    assert chart["hist"][-1] == 0.75


def test_build_stock_chart_reindexes_indicator_series_to_close_dates() -> None:
    frame = _ohlcv(rows=10)
    close = frame["Close"]
    shifted = pd.Series([1.0, 2.0], index=close.index[-2:])

    chart = build_stock_chart(frame, close, indicators={"rsi": shifted})

    assert chart["rsi"][:-2] == [None] * 8
    assert chart["rsi"][-2:] == [1.0, 2.0]
