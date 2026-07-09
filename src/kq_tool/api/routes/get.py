"""GET route response builders for the local API."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from urllib.parse import parse_qs, urlparse

from ..params import parse_stock_params, parse_strategy_backtest_params

ResponseFactory = Callable[..., object]


def _query_float(query: Mapping[str, list[str]], key: str, default: float) -> float:
    try:
        return float(query.get(key, [default])[0])
    except (TypeError, ValueError):
        return default


def _query_int(query: Mapping[str, list[str]], key: str, default: int) -> int:
    try:
        return int(float(query.get(key, [default])[0]))
    except (TypeError, ValueError):
        return default


def _query_text(query: Mapping[str, list[str]], key: str, default: str = "") -> str:
    try:
        return str(query.get(key, [default])[0])
    except (TypeError, ValueError):
        return default


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
    if path == "/api/market_report":
        return make_response("json", services["market_report"]())
    if path == "/api/regime_ai":
        return make_response("json", services["regime_ai"]())
    if path == "/api/recommend_portfolio":
        return make_response("json", services["recommend_portfolio"](_query_text(query, "profile", "neutral")))
    if path == "/api/portfolio_orders" and "portfolio_orders" in services:
        return make_response(
            "json",
            services["portfolio_orders"](
                _query_float(query, "amount", 10000000.0),
                _query_float(query, "tc", 10.0),
                _query_float(query, "slip", 5.0),
                _query_text(query, "holdings", ""),
            ),
        )
    if path == "/api/return_heatmap" and "return_heatmap" in services:
        return make_response(
            "json",
            services["return_heatmap"](_query_int(query, "limit", 36)),
        )
    if path == "/api/reco_track" and "reco_track" in services:
        return make_response(
            "json",
            services["reco_track"](_query_int(query, "months", 36)),
        )
    return make_response("not_found", {"error": "not found"}, code=404)
