from __future__ import annotations

from kq_tool.backtest.costs import (
    apply_transaction_cost,
    bps_to_rate,
    equal_weights,
    portfolio_turnover,
    transaction_cost_rate,
)


def test_equal_weights_normalizes_selected_tickers() -> None:
    weights = equal_weights(["A", "B", "C"])

    assert round(sum(weights.values()), 10) == 1.0
    assert weights["A"] == weights["B"] == weights["C"]


def test_portfolio_turnover_counts_buy_and_sell_notional() -> None:
    previous = {"A": 0.5, "B": 0.5}
    target = {"B": 0.5, "C": 0.5}

    assert portfolio_turnover(previous, target) == 1.0
    assert portfolio_turnover({}, target) == 1.0
    assert portfolio_turnover(target, target) == 0.0


def test_transaction_cost_rate_combines_cost_and_slippage_bps() -> None:
    rate = transaction_cost_rate(
        {"A": 1.0},
        {"B": 1.0},
        cost_bps=10,
        slippage_bps=5,
    )

    assert rate == 2.0 * 0.0015
    assert bps_to_rate(25) == 0.0025


def test_apply_transaction_cost_compounds_after_period_return() -> None:
    assert round(apply_transaction_cost(0.10, 0.01), 6) == 0.089
