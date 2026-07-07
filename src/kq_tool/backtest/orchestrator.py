"""Backtest orchestration helpers."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from math import comb

import pandas as pd

from kq_tool.config import RISK_FREE_RATE
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
from kq_tool.backtest.selector import robo_filter_decision, select_for_backtest
from kq_tool.backtest.strategy_meta import (
    QUANT,
    QUANT_ROBO_FILTER,
    QUANT_S2,
    QUANT_S2_ROBO_FILTER,
    normalize_strategy_key,
)


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
    filter_audit_records: list[dict] = []
    universe_names = universe_names or {}
    strategy_key = normalize_strategy_key(strategy)

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

        filter_decision = None
        if strategy_key in {QUANT_ROBO_FILTER.key, QUANT_S2_ROBO_FILTER.key}:
            base_strategy = QUANT_S2.key if strategy_key == QUANT_S2_ROBO_FILTER.key else QUANT.key
            filter_decision = robo_filter_decision(
                hist,
                base_strategy=base_strategy,
                top_n=top_n,
                precomputed=precomputed,
                cur_date=current_date,
            )
            selected = list(filter_decision["selected"])
        else:
            selected = select_for_backtest(hist, strategy, top_n, precomputed, current_date)
        if not selected:
            continue

        period_prices = price_df.loc[current_date:next_date]
        period_return = equal_weight_period_return(period_prices, selected)
        if period_return is None:
            continue
        if filter_decision is not None:
            audit = _build_filter_audit_record(
                filter_decision,
                period_prices,
                current_date,
                next_date,
                universe_names,
            )
            if audit is not None:
                filter_audit_records.append(audit)

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
    strategy_metrics = perf_metrics(eq_series, freq, risk_free_rate=RISK_FREE_RATE)
    benchmark_metrics = {}
    excess = {}
    if bench_values:
        bench_series = pd.Series(bench_values, index=pd.to_datetime(bench_dates))
        benchmark_metrics = perf_metrics(bench_series, freq, risk_free_rate=RISK_FREE_RATE)
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
        "risk_free_rate": RISK_FREE_RATE,
        "returns_basis": "net_after_transaction_costs_and_slippage",
        "holdings": holdings_log[-6:],
        "n_rebalance": len(holdings_log),
        "costs": {
            "transaction_cost_bps": float(transaction_cost_bps),
            "slippage_bps": float(slippage_bps),
            "total_turnover": round(float(total_turnover), 4),
            "total_cost_rate": round(float(total_cost_rate), 6),
        },
        "filter_audit": _summarize_filter_audit(filter_audit_records),
    }


def _ticker_return(period_prices: pd.DataFrame, ticker: str) -> float | None:
    if ticker not in period_prices.columns:
        return None
    series = pd.to_numeric(period_prices[ticker], errors="coerce").dropna()
    if len(series) < 2 or series.iloc[0] <= 0:
        return None
    return float(series.iloc[-1] / series.iloc[0] - 1)


def _mean_next_return(period_prices: pd.DataFrame, tickers: list[str]) -> float | None:
    returns = [
        value
        for ticker in tickers
        if (value := _ticker_return(period_prices, ticker)) is not None
    ]
    if not returns:
        return None
    return float(sum(returns) / len(returns))


def _build_filter_audit_record(
    decision: Mapping[str, object],
    period_prices: pd.DataFrame,
    current_date: object,
    next_date: object,
    universe_names: Mapping[str, str],
) -> dict | None:
    excluded = list(decision.get("excluded", []) or [])
    replacements = list(decision.get("replacements", []) or [])
    if not excluded and not replacements:
        return None
    excluded_return = _mean_next_return(period_prices, excluded)
    replacement_return = _mean_next_return(period_prices, replacements)
    spread = None
    hit = None
    if excluded_return is not None and replacement_return is not None:
        spread = replacement_return - excluded_return
        hit = replacement_return > excluded_return
    scores = decision.get("scores", {}) or {}
    return {
        "date": pd.Timestamp(current_date).strftime("%Y-%m"),
        "next_date": pd.Timestamp(next_date).strftime("%Y-%m"),
        "excluded": [universe_names.get(ticker, ticker) for ticker in excluded],
        "replacements": [universe_names.get(ticker, ticker) for ticker in replacements],
        "excluded_count": len(excluded),
        "replacement_count": len(replacements),
        "excluded_next_return": excluded_return,
        "replacement_next_return": replacement_return,
        "spread": spread,
        "hit": hit,
        "excluded_avg_score": _average_scores(excluded, scores),
        "replacement_avg_score": _average_scores(replacements, scores),
    }


def _average_scores(tickers: list[str], scores: Mapping[str, object]) -> float | None:
    values = [
        float(scores[ticker])
        for ticker in tickers
        if ticker in scores and isinstance(scores[ticker], (int, float))
    ]
    if not values:
        return None
    return float(sum(values) / len(values))


def _mean(values: list[float]) -> float | None:
    return float(sum(values) / len(values)) if values else None


def _summarize_filter_audit(records: list[dict]) -> dict:
    if not records:
        return {
            "n_events": 0,
            "n_excluded": 0,
            "n_replacements": 0,
            "avg_excluded_next_return": None,
            "avg_replacement_next_return": None,
            "avg_spread": None,
            "hit_count": 0,
            "hit_n": 0,
            "hit_rate": None,
            "hit_p_value": None,
            "recent": [],
        }
    excluded_returns = [
        float(record["excluded_next_return"])
        for record in records
        if record.get("excluded_next_return") is not None
    ]
    replacement_returns = [
        float(record["replacement_next_return"])
        for record in records
        if record.get("replacement_next_return") is not None
    ]
    spreads = [float(record["spread"]) for record in records if record.get("spread") is not None]
    hits = [bool(record["hit"]) for record in records if record.get("hit") is not None]
    hit_count = sum(hits)
    hit_n = len(hits)
    return {
        "n_events": len(records),
        "n_excluded": int(sum(record.get("excluded_count", 0) for record in records)),
        "n_replacements": int(sum(record.get("replacement_count", 0) for record in records)),
        "avg_excluded_next_return": _round_pct(_mean(excluded_returns)),
        "avg_replacement_next_return": _round_pct(_mean(replacement_returns)),
        "avg_spread": _round_pct(_mean(spreads)),
        "hit_count": int(hit_count),
        "hit_n": int(hit_n),
        "hit_rate": round(hit_count / hit_n * 100, 1) if hit_n else None,
        "hit_p_value": _binomial_two_sided_pvalue(hit_count, hit_n),
        "recent": records[-6:],
    }


def _binomial_two_sided_pvalue(successes: int, n: int) -> float | None:
    """Exact two-sided binomial test against p=0.5 for event-level hit rates."""

    if n <= 0:
        return None
    lower = sum(comb(n, k) for k in range(0, successes + 1)) / (2 ** n)
    upper = sum(comb(n, k) for k in range(successes, n + 1)) / (2 ** n)
    return round(float(min(1.0, 2 * min(lower, upper))), 4)


def _round_pct(value: float | None) -> float | None:
    if value is None:
        return None
    return round(float(value) * 100, 2)


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

