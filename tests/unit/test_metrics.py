from __future__ import annotations

import pandas as pd

from kq_tool.backtest.metrics import perf_metrics


def test_perf_metrics_returns_empty_for_short_series() -> None:
    equity = pd.Series([100, 101, 102])

    assert perf_metrics(equity) == {}


def test_perf_metrics_calculates_core_fields() -> None:
    equity = pd.Series(
        [100, 105, 103, 110, 115, 120],
        index=pd.date_range("2024-01-31", periods=6, freq="ME"),
    )

    result = perf_metrics(equity, freq_per_year=12, risk_free_rate=0.0)

    assert set(result) == {
        "total_return",
        "cagr",
        "vol",
        "sharpe",
        "mdd",
        "calmar",
        "win_rate",
        "years",
    }
    assert result["total_return"] == 20.0
    assert result["mdd"] < 0


def test_perf_metrics_win_rate_uses_period_returns() -> None:
    equity = pd.Series([100, 110, 99, 108.9, 98.01])

    result = perf_metrics(equity, freq_per_year=4, risk_free_rate=0.0)

    assert result["win_rate"] == 50.0
