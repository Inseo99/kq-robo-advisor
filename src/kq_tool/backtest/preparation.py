"""Backtest data preparation helpers."""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd


DEFAULT_BACKTEST_START = pd.Timestamp("2014-01-01")
ROLLING_BACKTEST_PERIODS = ("1y", "2y", "3y", "5y", "7y")
FIXED_START_PERIODS = ("12y", "15y", "20y", "max", None)


def use_fixed_start_for_period(period: str | None) -> bool:
    """Return whether a strategy backtest period should use the fixed start."""

    return period in FIXED_START_PERIODS or period not in ROLLING_BACKTEST_PERIODS


def evaluation_start_for_period(
    price_df: pd.DataFrame,
    period: str | None,
    *,
    period_days_fn: Callable[[str | None, int], int] | None = None,
    backtest_start: pd.Timestamp = DEFAULT_BACKTEST_START,
) -> pd.Timestamp:
    """Return the first date that belongs to the reported backtest window."""

    if price_df is None or price_df.empty or use_fixed_start_for_period(period):
        return backtest_start
    days = period_days_fn(period, 1095) if period_days_fn is not None else 1095
    return price_df.index[-1] - pd.Timedelta(days=days)


def filter_strategy_price_period(
    price_df: pd.DataFrame,
    period: str | None,
    *,
    period_days_fn: Callable[[str | None, int], int] | None = None,
    backtest_start: pd.Timestamp = DEFAULT_BACKTEST_START,
    warmup_days: int = 0,
) -> pd.DataFrame:
    """Filter a strategy frame, keeping optional lookback warm-up rows."""

    start = evaluation_start_for_period(
        price_df,
        period,
        period_days_fn=period_days_fn,
        backtest_start=backtest_start,
    )
    if not use_fixed_start_for_period(period) and warmup_days > 0:
        start = start - pd.Timedelta(days=int(warmup_days))
    return price_df[price_df.index >= start]


def filter_strategy_benchmark_period(
    benchmark: pd.Series | None,
    period: str | None,
    *,
    backtest_start: pd.Timestamp = DEFAULT_BACKTEST_START,
) -> pd.Series | None:
    """Apply the fixed-start benchmark filter used by legacy strategy tests."""

    if benchmark is None:
        return None
    if use_fixed_start_for_period(period):
        return benchmark[benchmark.index >= backtest_start]
    return benchmark


def prepare_strategy_price_frame(
    price_df: pd.DataFrame,
    period: str | None,
    *,
    period_days_fn: Callable[[str | None, int], int] | None = None,
    min_observations: int = 60,
    backtest_start: pd.Timestamp = DEFAULT_BACKTEST_START,
    warmup_days: int = 0,
) -> tuple[pd.DataFrame | None, str | None]:
    """Filter, validate, and forward-fill price data for strategy backtests."""

    if price_df is None or price_df.empty:
        return None, "데이터 없음"

    filtered = filter_strategy_price_period(
        price_df,
        period,
        period_days_fn=period_days_fn,
        backtest_start=backtest_start,
        warmup_days=warmup_days,
    )

    valid_cols = [col for col in filtered.columns if filtered[col].notna().sum() >= min_observations]
    if not valid_cols:
        return None, "유효 종목 부족"

    prepared = filtered[valid_cols].ffill().dropna(how="all")
    if prepared.empty:
        return None, "데이터 없음"
    return prepared, None


_use_fixed_start_for_period = use_fixed_start_for_period
_prepare_strategy_price_frame = prepare_strategy_price_frame
