"""Asset-allocation strategy evaluation helpers."""

from __future__ import annotations

from collections.abc import Callable, Mapping

import numpy as np
import pandas as pd

from kq_tool.analyzer.indicators import close_series
from kq_tool.portfolio.risk_based import diversification_ratio, risk_contributions


RiskContributionsFn = Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, float]]
DiversificationRatioFn = Callable[[np.ndarray, np.ndarray, np.ndarray], float]
CloseFn = Callable[[pd.DataFrame | pd.Series], pd.Series]
WeightsFn = Callable[..., np.ndarray]


def inverse_volatility_weights(
    price_frames: Mapping[str, pd.DataFrame],
    *,
    tickers: list[str] | tuple[str, ...] | None = None,
    close_fn: CloseFn = close_series,
    lookback: int = 60,
    floor: float = 1e-5,
    fallback_vol: float = 0.01,
) -> dict[str, float]:
    """Build inverse-volatility ETF weights from recent daily returns."""

    selected_tickers = list(tickers or price_frames.keys())
    vols: dict[str, float] = {}
    for ticker in selected_tickers:
        try:
            close = close_fn(price_frames[ticker])
            vol = float(close.pct_change().dropna().tail(lookback).std())
            vols[ticker] = max(vol, floor)
        except Exception:
            vols[ticker] = fallback_vol

    inv = {ticker: 1 / vol for ticker, vol in vols.items() if vol > 0}
    total = float(sum(inv.values()))
    if total <= 0:
        return {}
    return {ticker: round(float(value / total), 4) for ticker, value in inv.items()}


def gtaa_weights(
    price_frames: Mapping[str, pd.DataFrame],
    *,
    tickers: list[str] | tuple[str, ...] | None = None,
    close_fn: CloseFn = close_series,
    ma_window: int = 200,
    fallback_count: int = 3,
) -> dict[str, float]:
    """Build a simple GTAA allocation from price above long moving average."""

    selected_tickers = list(tickers or price_frames.keys())
    active: list[str] = []
    for ticker in selected_tickers:
        frame = price_frames.get(ticker)
        if frame is None:
            continue
        try:
            close = close_fn(frame)
            if len(close) >= ma_window and float(close.iloc[-1]) > float(close.rolling(ma_window).mean().iloc[-1]):
                active.append(ticker)
        except Exception:
            continue

    if not active:
        active = selected_tickers[:fallback_count]
    if not active:
        return {}
    weight = round(1 / len(active), 4)
    return {ticker: weight for ticker in active}


def risk_based_strategy_weights(
    monthly_returns: pd.DataFrame,
    *,
    gmv_fn: WeightsFn,
    mdp_fn: WeightsFn,
    erc_fn: WeightsFn,
) -> dict[str, dict[str, float] | None]:
    """Build GMV, MDP, and ERC weights from monthly return covariance."""

    tickers = list(monthly_returns.columns)
    cov = monthly_returns[tickers].cov().values * 12.0
    vols = np.sqrt(np.diag(cov))
    results: dict[str, dict[str, float] | None] = {}

    try:
        weights = gmv_fn(cov)
        results["GMV"] = {ticker: round(float(weight), 4) for ticker, weight in zip(tickers, weights)}
    except Exception:
        results["GMV"] = None

    try:
        weights = mdp_fn(cov, vols)
        results["MDP"] = {ticker: round(float(weight), 4) for ticker, weight in zip(tickers, weights)}
    except Exception:
        results["MDP"] = None

    try:
        weights = erc_fn(cov)
        results["ERC"] = {ticker: round(float(weight), 4) for ticker, weight in zip(tickers, weights)}
    except Exception:
        results["ERC"] = None

    return results


def evaluate_asset_allocation_strategies(
    monthly_returns: pd.DataFrame,
    strategies: Mapping[str, Mapping[str, float] | None],
    *,
    risk_contributions_fn: RiskContributionsFn = risk_contributions,
    diversification_ratio_fn: DiversificationRatioFn = diversification_ratio,
    risk_based_keys: list[str] | tuple[str, ...] = (),
    risk_free_rate: float = 0.035,
) -> dict[str, dict]:
    """Evaluate ETF allocation strategies from monthly return data."""

    results: dict[str, dict] = {}
    if monthly_returns is None or monthly_returns.empty:
        return results

    for name, weights in strategies.items():
        if not weights:
            continue
        try:
            available = [
                ticker
                for ticker in weights
                if ticker in monthly_returns.columns and weights.get(ticker, 0) > 0
            ]
            if not available:
                continue

            weight_vector = np.array([weights[ticker] for ticker in available], dtype=float)
            total_weight = float(weight_vector.sum())
            if total_weight <= 0:
                continue
            weight_vector /= total_weight

            portfolio_returns = monthly_returns[available].dot(weight_vector)
            count = len(portfolio_returns)
            if count == 0:
                continue

            years = count / 12
            cumulative_return = float((1 + portfolio_returns).prod())
            cagr = cumulative_return ** (1 / years) - 1 if years > 0 else 0.0
            volatility = float(portfolio_returns.std() * np.sqrt(12))
            sharpe = (cagr - risk_free_rate) / volatility if volatility > 0 else 0.0
            wealth = (1 + portfolio_returns).cumprod()
            drawdown = (wealth - wealth.cummax()) / wealth.cummax()
            max_drawdown = float(drawdown.min())
            cumulative_series = wealth * 100

            sub_cov = monthly_returns[available].cov().values * 12.0
            contributions, portfolio_sigma = risk_contributions_fn(weight_vector, sub_cov)
            contribution_pct = (
                contributions / portfolio_sigma * 100.0
                if portfolio_sigma > 1e-12
                else np.zeros_like(contributions)
            )
            div_ratio = diversification_ratio_fn(
                weight_vector,
                sub_cov,
                np.sqrt(np.diag(sub_cov)),
            )

            results[name] = {
                "weights": {ticker: round(float(weights[ticker]), 4) for ticker in available},
                "metrics": {
                    "cagr": round(float(cagr) * 100, 2),
                    "vol": round(float(volatility) * 100, 2),
                    "sharpe": round(float(sharpe), 3),
                    "mdd": round(float(max_drawdown) * 100, 2),
                    "calmar": round(float(cagr / abs(max_drawdown)), 3)
                    if abs(max_drawdown) > 1e-6
                    else 0,
                },
                "cum": [round(float(value), 2) for value in cumulative_series.tolist()],
                "dates": [date.strftime("%Y-%m") for date in cumulative_series.index],
                "risk_contrib": {
                    ticker: round(float(percent), 2)
                    for ticker, percent in zip(available, contribution_pct)
                },
                "capital_weight": {
                    ticker: round(float(weight) * 100, 2)
                    for ticker, weight in zip(available, weight_vector)
                },
                "diversification_ratio": round(float(div_ratio), 3),
                "is_risk_based": name in risk_based_keys,
            }
        except Exception:
            continue

    return results


_evaluate_asset_allocation_strategies = evaluate_asset_allocation_strategies
_inverse_volatility_weights = inverse_volatility_weights
_gtaa_weights = gtaa_weights
_risk_based_strategy_weights = risk_based_strategy_weights
