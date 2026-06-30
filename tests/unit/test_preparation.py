from __future__ import annotations

import numpy as np
import pandas as pd

from kq_tool.backtest.preparation import (
    filter_strategy_benchmark_period,
    prepare_strategy_price_frame,
    use_fixed_start_for_period,
)


def _prices(rows: int = 100) -> pd.DataFrame:
    index = pd.date_range("2023-01-02", periods=rows, freq="B")
    return pd.DataFrame(
        {
            "A": np.linspace(100, 150, rows),
            "B": [np.nan] * 30 + list(np.linspace(50, 80, rows - 30)),
            "C": [np.nan] * rows,
        },
        index=index,
    )


def test_use_fixed_start_for_period_matches_strategy_policy() -> None:
    assert use_fixed_start_for_period("max") is True
    assert use_fixed_start_for_period("12y") is True
    assert use_fixed_start_for_period(None) is True
    assert use_fixed_start_for_period("3y") is False
    assert use_fixed_start_for_period("unknown") is True


def test_prepare_strategy_price_frame_filters_valid_columns_and_preserves_rows() -> None:
    prepared, error = prepare_strategy_price_frame(
        _prices(),
        "3y",
        period_days_fn=lambda _period, _default: 3650,
        min_observations=60,
    )

    assert error is None
    assert prepared is not None
    assert list(prepared.columns) == ["A", "B"]
    assert prepared.index[0] == _prices().index[0]
    assert prepared["B"].notna().sum() == 70


def test_prepare_strategy_price_frame_returns_error_when_no_valid_columns() -> None:
    frame = pd.DataFrame({"A": [np.nan, np.nan]}, index=pd.date_range("2024-01-01", periods=2))

    prepared, error = prepare_strategy_price_frame(frame, "max")

    assert prepared is None
    assert error == "유효 종목 부족"


def test_filter_strategy_benchmark_period_uses_fixed_start_only() -> None:
    index = pd.to_datetime(["2013-12-31", "2014-01-02", "2014-01-03"])
    benchmark = pd.Series([90, 100, 101], index=index)

    filtered = filter_strategy_benchmark_period(benchmark, "max")
    rolling = filter_strategy_benchmark_period(benchmark, "3y")

    assert filtered.index[0] == pd.Timestamp("2014-01-02")
    assert rolling.equals(benchmark)


def test_server_reuses_backtest_preparation_helpers() -> None:
    import server
    from kq_tool.backtest.preparation import prepare_strategy_price_frame as helper

    assert server._kq_prepare_strategy_price_frame is helper
