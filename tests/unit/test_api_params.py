from __future__ import annotations

from kq_tool.api.params import (
    first_query_value,
    parse_stock_params,
    parse_strategy_backtest_params,
)


def test_first_query_value_matches_parse_qs_first_value_semantics() -> None:
    assert first_query_value({"x": ["a", "b"]}, "x", "fallback") == "a"
    assert first_query_value({"x": []}, "x", "fallback") == "fallback"
    assert first_query_value({}, "x", "fallback") == "fallback"


def test_parse_stock_params_trims_ticker_and_defaults_period() -> None:
    params = parse_stock_params({"t": [" 000660.KS "]})

    assert params.ticker == "000660.KS"
    assert params.period == "1y"


def test_parse_strategy_backtest_params_includes_trading_frictions() -> None:
    params = parse_strategy_backtest_params(
        {"s": ["robo"], "n": ["7"], "r": ["Q"], "p": ["5y"], "tc": ["10"], "slip": ["5"]}
    )

    assert params.strategy == "robo"
    assert params.top_n == 7
    assert params.rebalance == "Q"
    assert params.period == "5y"
    assert params.transaction_cost_bps == 10.0
    assert params.slippage_bps == 5.0


def test_parse_strategy_backtest_params_defaults_to_standard_trading_costs() -> None:
    params = parse_strategy_backtest_params({})

    assert params.strategy == "quant"
    assert params.top_n == 5
    assert params.rebalance == "M"
    assert params.period == "3y"
    assert params.transaction_cost_bps == 10.0
    assert params.slippage_bps == 5.0


def test_server_keeps_param_parsing_inside_api_action_helpers() -> None:
    import server

    assert not hasattr(server, "_kq_parse_stock_params")
    assert not hasattr(server, "_kq_parse_strategy_backtest_params")

