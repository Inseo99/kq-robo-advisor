"""Backtest orchestration helpers."""

from __future__ import annotations

from collections.abc import Callable, Mapping

import pandas as pd

from kq_tool.backtest.costs import (
    apply_transaction_cost,
    equal_weights,
    portfolio_turnover,
    transaction_cost_rate,
)
from kq_tool.backtest.engine import (
    equal_weight_period_return,
    normalize_benchmark_to_equity,
    underwater_curve,
)
from kq_tool.backtest.metrics import perf_metrics
from kq_tool.backtest.selector import select_for_backtest


PitSelector = Callable[[object], list[str]]

def actual_rebalance_dates(price_df: pd.DataFrame, rule: str) -> pd.DatetimeIndex:
    """Return actual last observed dates for each resample bucket."""

    labels = price_df.resample(rule).last().index
    dates = []
    for label in labels:
        eligible = price_df.index[price_df.index <= label]
        if len(eligible):
            date = eligible[-1]
            if not dates or dates[-1] != date:
                dates.append(date)
    return pd.DatetimeIndex(dates)




def run_rebalanced_strategy_backtest(
    price_df: pd.DataFrame,
    *,
    strategy: str,
    top_n: int,
    rebalance: str,
    period: str,
    benchmark: pd.Series | None = None,
    precomputed: dict[str, pd.DataFrame] | None = None,
    pit_selector: PitSelector | None = None,
    universe_names: Mapping[str, str] | None = None,
    transaction_cost_bps: float = 0.0,
    slippage_bps: float = 0.0,
    evaluation_start: object | None = None,
) -> dict:
    """Run the reusable strategy backtest loop on prepared price data."""

    if price_df.empty:
        return {"error": "데이터 없음"}
    valid_cols = [col for col in price_df.columns if price_df[col].notna().sum() >= 60]
    if not valid_cols:
        return {"error": "유효 종목 부족"}

    price_df = price_df[valid_cols].ffill().dropna(how="all")
    if price_df.empty:
        return {"error": "데이터 없음"}

    rule = {"M": "ME", "Q": "QE", "W": "W"}.get(rebalance, "ME")
    rebal_dates = actual_rebalance_dates(price_df, rule)
    months = price_df.loc[rebal_dates]
    if len(months) < 2:
        return {"error": "리밸런싱 기간 부족"}

    equity = [100.0]
    eq_dates: list[str] = []
    evaluation_ts = pd.Timestamp(evaluation_start) if evaluation_start is not None else None
    holdings_log = []
    previous_weights: dict[str, float] = {}
    total_cost_rate = 0.0
    total_turnover = 0.0
    universe_names = universe_names or {}

    for index in range(len(months) - 1):
        current_date = months.index[index]
        next_date = months.index[index + 1]
        if evaluation_ts is not None and current_date < evaluation_ts:
            continue
        hist = price_df.loc[:current_date]
        if len(hist) < 60:
            continue

        if pit_selector is not None:
            pit_available = [ticker for ticker in pit_selector(current_date) if ticker in hist.columns]
            if len(pit_available) < top_n:
                continue
            hist = hist[pit_available]

        selected = select_for_backtest(hist, strategy, top_n, precomputed, current_date)
        if not selected:
            continue

        period_prices = price_df.loc[current_date:next_date]
        period_return = equal_weight_period_return(period_prices, selected)
        if period_return is None:
            continue

        target_weights = equal_weights(selected)
        turnover = portfolio_turnover(previous_weights, target_weights)
        cost_rate = transaction_cost_rate(
            previous_weights,
            target_weights,
            cost_bps=transaction_cost_bps,
            slippage_bps=slippage_bps,
        )
        net_period_return = apply_transaction_cost(period_return, cost_rate)

        if not eq_dates:
            eq_dates.append(current_date.strftime("%Y-%m-%d"))
        equity.append(equity[-1] * (1 + net_period_return))
        eq_dates.append(next_date.strftime("%Y-%m-%d"))
        holdings_log.append(
            {
                "date": current_date.strftime("%Y-%m"),
                "tickers": [universe_names.get(ticker, ticker) for ticker in selected],
                "turnover": round(float(turnover), 4),
                "cost_rate": round(float(cost_rate), 6),
            }
        )
        previous_weights = target_weights
        total_turnover += turnover
        total_cost_rate += cost_rate

    if not eq_dates:
        anchor = evaluation_ts if evaluation_ts is not None else price_df.index[0]
        eligible = price_df.index[price_df.index >= anchor]
        anchor = eligible[0] if len(eligible) else price_df.index[-1]
        eq_dates = [anchor.strftime("%Y-%m-%d")]
    eq_series = pd.Series(equity, index=pd.to_datetime(eq_dates))
    bench_values, bench_dates = [], []
    if benchmark is not None:
        bench_values, bench_dates = normalize_benchmark_to_equity(benchmark, eq_series.index)

    freq = {"M": 12, "Q": 4, "W": 52}.get(rebalance, 12)
    strategy_metrics = perf_metrics(eq_series, freq)
    benchmark_metrics = {}
    excess = {}
    if bench_values:
        bench_series = pd.Series(bench_values, index=pd.to_datetime(bench_dates))
        benchmark_metrics = perf_metrics(bench_series, freq)
        excess = _excess_metrics(eq_series, bench_series, strategy_metrics, benchmark_metrics)

    return {
        "strategy": strategy,
        "top_n": top_n,
        "rebalance": rebalance,
        "period": period,
        "equity": [round(float(value), 2) for value in eq_series.tolist()],
        "dates": [date.strftime("%Y-%m-%d") for date in eq_series.index],
        "benchmark": bench_values,
        "benchmark_dates": bench_dates,
        "underwater": underwater_curve(eq_series),
        "metrics": strategy_metrics,
        "bench_metrics": benchmark_metrics,
        "excess": excess,
        "holdings": holdings_log[-6:],
        "n_rebalance": len(holdings_log),
        "costs": {
            "transaction_cost_bps": float(transaction_cost_bps),
            "slippage_bps": float(slippage_bps),
            "total_turnover": round(float(total_turnover), 4),
            "total_cost_rate": round(float(total_cost_rate), 6),
        },
    }


def _excess_metrics(
    equity: pd.Series,
    benchmark: pd.Series,
    strategy_metrics: Mapping[str, float],
    benchmark_metrics: Mapping[str, float],
) -> dict:
    strategy_returns = equity.pct_change().dropna()
    benchmark_returns = benchmark.pct_change().dropna()
    common = strategy_returns.index.intersection(benchmark_returns.index)
    if len(common) <= 5:
        return {}
    strategy_common = strategy_returns.loc[common]
    benchmark_common = benchmark_returns.loc[common]
    variance = float(benchmark_common.var())
    beta = float(strategy_common.cov(benchmark_common) / variance) if variance > 0 else 1.0
    alpha_ann = strategy_metrics.get("cagr", 0) - benchmark_metrics.get("cagr", 0)
    excess_return = strategy_metrics.get("total_return", 0) - benchmark_metrics.get("total_return", 0)
    return {
        "beta": round(beta, 3),
        "alpha": round(float(alpha_ann), 2),
        "excess_return": round(float(excess_return), 2),
    }


_run_rebalanced_strategy_backtest = run_rebalanced_strategy_backtest

