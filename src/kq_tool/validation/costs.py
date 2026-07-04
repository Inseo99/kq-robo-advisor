"""Cost helpers for signal-validation experiments."""

from __future__ import annotations

from collections.abc import Iterable

from kq_tool.backtest.costs import bps_to_rate


def signal_trade_cost_rate(
    *,
    cost_bps: float = 0.0,
    slippage_bps: float = 0.0,
    trade_sides: int = 2,
) -> float:
    """Return per-signal validation cost as a decimal return haircut."""

    sides = max(0, int(trade_sides))
    total_bps = max(0.0, float(cost_bps)) + max(0.0, float(slippage_bps))
    return bps_to_rate(total_bps) * sides


def net_edge_return(
    gross_return: float | None,
    *,
    cost_bps: float = 0.0,
    slippage_bps: float = 0.0,
    trade_sides: int = 2,
) -> float | None:
    """Apply a simple cost haircut to a direction-adjusted signal edge."""

    if gross_return is None:
        return None
    return float(gross_return) - signal_trade_cost_rate(
        cost_bps=cost_bps,
        slippage_bps=slippage_bps,
        trade_sides=trade_sides,
    )


def apply_signal_costs(
    returns: Iterable[float],
    *,
    cost_bps: float = 0.0,
    slippage_bps: float = 0.0,
    trade_sides: int = 2,
) -> list[float]:
    """Apply the same per-event cost haircut to a return collection."""

    cost = signal_trade_cost_rate(
        cost_bps=cost_bps,
        slippage_bps=slippage_bps,
        trade_sides=trade_sides,
    )
    return [float(value) - cost for value in returns]
