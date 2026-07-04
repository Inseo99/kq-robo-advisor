"""Transaction cost and slippage helpers for backtests."""

from __future__ import annotations

from collections.abc import Mapping, Sequence


def equal_weights(tickers: Sequence[str]) -> dict[str, float]:
    """Return equal weights for a selected ticker list."""

    clean = [ticker for ticker in tickers if ticker]
    if not clean:
        return {}
    weight = 1.0 / len(clean)
    return {ticker: weight for ticker in clean}


def portfolio_turnover(
    previous_weights: Mapping[str, float] | None,
    target_weights: Mapping[str, float] | None,
) -> float:
    """Return traded notional as a fraction of portfolio value."""

    previous_weights = previous_weights or {}
    target_weights = target_weights or {}
    tickers = set(previous_weights).union(target_weights)
    return float(
        sum(abs(float(target_weights.get(ticker, 0.0)) - float(previous_weights.get(ticker, 0.0))) for ticker in tickers)
    )


def bps_to_rate(bps: float) -> float:
    """Convert basis points to decimal rate."""

    return float(bps) / 10_000.0


def transaction_cost_rate(
    previous_weights: Mapping[str, float] | None,
    target_weights: Mapping[str, float] | None,
    *,
    cost_bps: float = 0.0,
    slippage_bps: float = 0.0,
) -> float:
    """Return portfolio-level cost rate for a rebalance."""

    total_bps = max(0.0, float(cost_bps)) + max(0.0, float(slippage_bps))
    if total_bps <= 0:
        return 0.0
    return portfolio_turnover(previous_weights, target_weights) * bps_to_rate(total_bps)


def apply_transaction_cost(period_return: float, cost_rate: float) -> float:
    """Apply transaction cost after gross period return."""

    return float((1.0 + float(period_return)) * (1.0 - max(0.0, float(cost_rate))) - 1.0)
