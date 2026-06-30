"""Reusable backtest engine helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd


def equal_weight_period_return(period_prices: pd.DataFrame, selected: list[str]) -> float | None:
    """Return equal-weight return over one rebalance period.

    Missing tickers are skipped and the remaining weights are rescaled to keep
    the legacy behavior.
    """

    if len(period_prices) < 2 or not selected:
        return None

    base_weight = {ticker: 1 / len(selected) for ticker in selected}
    period_return = 0.0
    valid_count = 0

    for ticker in selected:
        try:
            series = period_prices[ticker].dropna()
            if len(series) < 2:
                continue
            p0, p1 = float(series.iloc[0]), float(series.iloc[-1])
            if p0 <= 0 or not np.isfinite(p0) or not np.isfinite(p1):
                continue
            period_return += base_weight[ticker] * ((p1 - p0) / p0)
            valid_count += 1
        except Exception:
            continue

    if valid_count == 0:
        return None
    if valid_count < len(selected):
        period_return = period_return * len(selected) / valid_count
    return float(period_return)


def underwater_curve(equity: pd.Series) -> list[float]:
    """Return drawdown/underwater curve in percentage points."""

    peak = equity.cummax()
    return [round(float((equity.iloc[i] - peak.iloc[i]) / peak.iloc[i]) * 100, 2) for i in range(len(equity))]


def normalize_benchmark_to_equity(benchmark: pd.Series, equity_index: pd.Index) -> tuple[list[float], list[str]]:
    """Align and normalize benchmark to an equity curve index."""

    aligned = benchmark.reindex(equity_index, method="ffill").dropna()
    if len(aligned) <= 1:
        return [], []
    normalized = aligned / aligned.iloc[0] * 100
    values = [round(float(value), 2) for value in normalized.tolist()]
    dates = [date.strftime("%Y-%m-%d") for date in normalized.index]
    return values, dates
