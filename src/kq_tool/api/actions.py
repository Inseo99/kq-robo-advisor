"""Endpoint action helpers for the local API server."""

from __future__ import annotations

from collections.abc import Callable

from .params import QueryParams, parse_stock_params, parse_strategy_backtest_params

JsonAction = Callable[[Callable[[], object]], object]


def run_json_service_action(
    service: Callable[[], object],
    *,
    json_action: JsonAction,
) -> object:
    """Run a zero-argument service through the shared JSON action wrapper."""

    return json_action(service)

def run_stock_action(
    query: QueryParams,
    *,
    analyze_stock: Callable[[str, str], object],
    json_action: JsonAction,
    parse_params: Callable[[QueryParams], object] = parse_stock_params,
) -> object:
    """Parse `/api/stock` params and run the stock analysis JSON action."""

    params = parse_params(query)
    return json_action(lambda: analyze_stock(params.ticker, params.period))


def run_strategy_backtest_action(
    query: QueryParams,
    *,
    run_strategy_backtest: Callable[[str, int, str, str, float, float], object],
    json_action: JsonAction,
    parse_params: Callable[[QueryParams], object] = parse_strategy_backtest_params,
) -> object:
    """Parse `/api/stratbt` params and run the strategy backtest JSON action."""

    params = parse_params(query)
    return json_action(
        lambda: run_strategy_backtest(
            params.strategy,
            params.top_n,
            params.rebalance,
            params.period,
            params.transaction_cost_bps,
            params.slippage_bps,
        )
    )

