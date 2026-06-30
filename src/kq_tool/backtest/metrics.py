"""Backtest performance metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd


def perf_metrics(
    equity: pd.Series,
    freq_per_year: int = 12,
    risk_free_rate: float = 0.035,
) -> dict:
    """Convert an equity curve into performance metrics."""

    returns = equity.pct_change().dropna()
    n = len(returns)
    if n < 3:
        return {}

    years = n / freq_per_year
    total = float(equity.iloc[-1] / equity.iloc[0])
    cagr = total ** (1 / years) - 1 if years > 0 else 0
    vol = float(returns.std() * np.sqrt(freq_per_year))
    sharpe = (cagr - risk_free_rate) / vol if vol > 0 else 0
    peak = equity.cummax()
    drawdown = (equity - peak) / peak
    mdd = float(drawdown.min())
    calmar = cagr / abs(mdd) if abs(mdd) > 1e-6 else 0
    win_rate = float((returns > 0).sum() / n) if n > 0 else 0

    return {
        "total_return": round((total - 1) * 100, 2),
        "cagr": round(float(cagr) * 100, 2),
        "vol": round(vol * 100, 2),
        "sharpe": round(float(sharpe), 3),
        "mdd": round(mdd * 100, 2),
        "calmar": round(float(calmar), 3),
        "win_rate": round(win_rate * 100, 1),
        "years": round(years, 2),
    }


_perf_metrics = perf_metrics
