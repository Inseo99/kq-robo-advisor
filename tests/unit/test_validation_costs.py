from __future__ import annotations

from kq_tool.validation.costs import apply_signal_costs, net_edge_return, signal_trade_cost_rate


def test_signal_trade_cost_rate_uses_round_trip_by_default() -> None:
    assert signal_trade_cost_rate(cost_bps=10, slippage_bps=5) == 0.003
    assert signal_trade_cost_rate(cost_bps=10, slippage_bps=5, trade_sides=1) == 0.0015


def test_net_edge_return_subtracts_cost_haircut() -> None:
    assert round(net_edge_return(0.02, cost_bps=10, slippage_bps=5), 6) == 0.017
    assert net_edge_return(None, cost_bps=10) is None


def test_apply_signal_costs_handles_collections() -> None:
    assert apply_signal_costs([0.01, -0.01], cost_bps=5, slippage_bps=5) == [0.008, -0.012]
