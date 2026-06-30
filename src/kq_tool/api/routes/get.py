"""GET route response builders for the local API."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from urllib.parse import parse_qs, urlparse

from ..params import parse_stock_params, parse_strategy_backtest_params

ResponseFactory = Callable[..., object]


def build_get_response(
    path_with_query: str,
    services: Mapping[str, Callable[..., object]],
    *,
    make_response: ResponseFactory,
) -> object:
    """Build a response object for a local GET request path."""

    parsed = urlparse(path_with_query)
    path = parsed.path
    query = parse_qs(parsed.query)

    if path == "/":
        return make_response("file", "index.html", content_type="text/html; charset=utf-8")
    if path == "/api/ping":
        return make_response("json", {"ok": True})
    if path == "/api/health":
        return make_response("json", services["health"]())
    if path == "/api/macro":
        return make_response("json", services["macro"]())
    if path == "/api/stock":
        params = parse_stock_params(query)
        return make_response("json", services["stock"](params.ticker, params.period))
    if path == "/api/screen":
        return make_response("json", services["screen"]())
    if path == "/api/backtest":
        return make_response("json", services["backtest"]())
    if path == "/api/stratbt":
        params = parse_strategy_backtest_params(query)
        return make_response(
            "json",
            services["stratbt"](
                params.strategy,
                params.top_n,
                params.rebalance,
                params.period,
                params.transaction_cost_bps,
                params.slippage_bps,
            ),
        )
    if path == "/api/regime_ai":
        return make_response("json", services["regime_ai"]())
    if path == "/api/recommend_portfolio":
        return make_response("json", services["recommend_portfolio"]())
    return make_response("not_found", {"error": "not found"}, code=404)
