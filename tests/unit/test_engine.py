from __future__ import annotations

import pandas as pd

from kq_tool.backtest.engine import (
    equal_weight_period_return,
    normalize_benchmark_to_equity,
    underwater_curve,
)


def test_equal_weight_period_return_rescales_missing_tickers() -> None:
    prices = pd.DataFrame({"A": [100, 110], "B": [None, None]})

    result = equal_weight_period_return(prices, ["A", "B"])

    assert result == 0.1


def test_equal_weight_period_return_returns_none_without_valid_prices() -> None:
    prices = pd.DataFrame({"A": [None, None]})

    assert equal_weight_period_return(prices, ["A"]) is None


def test_underwater_curve_returns_drawdowns() -> None:
    equity = pd.Series([100, 110, 99, 120])

    assert underwater_curve(equity) == [0.0, 0.0, -10.0, 0.0]


def test_normalize_benchmark_to_equity_aligns_dates() -> None:
    index = pd.to_datetime(["2024-01-31", "2024-02-29", "2024-03-31"])
    benchmark = pd.Series([100, 110, 121], index=index)

    values, dates = normalize_benchmark_to_equity(benchmark, index)

    assert values == [100.0, 110.0, 121.0]
    assert dates == ["2024-01-31", "2024-02-29", "2024-03-31"]
