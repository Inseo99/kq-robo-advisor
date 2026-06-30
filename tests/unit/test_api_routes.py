from __future__ import annotations

from kq_tool.api.routes import build_get_response
from kq_tool.backtest.strategy_meta import QUANT_COMPARE


def _make_response(kind, payload, code=200, content_type=None):
    return {
        "kind": kind,
        "payload": payload,
        "code": code,
        "content_type": content_type,
    }


def _services():
    return {
        "macro": lambda: {"macro": True},
        "health": lambda: {"ok": True},
        "stock": lambda ticker, period: {"ticker": ticker, "period": period},
        "screen": lambda: {"screen": True},
        "backtest": lambda: {"backtest": True},
        "stratbt": lambda strategy, top_n, rebalance, period, transaction_cost_bps=0.0, slippage_bps=0.0: {
            "strategy": strategy,
            "top_n": top_n,
            "rebalance": rebalance,
            "period": period,
            "transaction_cost_bps": transaction_cost_bps,
            "slippage_bps": slippage_bps,
        },
        "regime_ai": lambda: {"regime": True},
        "recommend_portfolio": lambda: {"recommend": True},
    }


def test_build_get_response_root_returns_index_file_response() -> None:
    response = build_get_response("/", _services(), make_response=_make_response)

    assert response == {
        "kind": "file",
        "payload": "index.html",
        "code": 200,
        "content_type": "text/html; charset=utf-8",
    }


def test_build_get_response_parses_stock_query() -> None:
    response = build_get_response("/api/stock?t=000660.KS&p=max", _services(), make_response=_make_response)

    assert response["kind"] == "json"
    assert response["payload"] == {"ticker": "000660.KS", "period": "max"}


def test_build_get_response_parses_strategy_backtest_costs() -> None:
    response = build_get_response(
        "/api/stratbt?s=quant_compare&n=3&r=Q&p=12y&tc=10&slip=5",
        _services(),
        make_response=_make_response,
    )

    assert response["payload"] == {
        "strategy": QUANT_COMPARE.key,
        "top_n": 3,
        "rebalance": "Q",
        "period": "12y",
        "transaction_cost_bps": 10.0,
        "slippage_bps": 5.0,
    }


def test_build_get_response_routes_core_zero_arg_services() -> None:
    services = _services()

    assert build_get_response("/api/ping", services, make_response=_make_response)["payload"] == {"ok": True}
    assert build_get_response("/api/health", services, make_response=_make_response)["payload"] == {"ok": True}
    assert build_get_response("/api/macro", services, make_response=_make_response)["payload"] == {"macro": True}
    assert build_get_response("/api/screen", services, make_response=_make_response)["payload"] == {"screen": True}
    assert build_get_response("/api/backtest", services, make_response=_make_response)["payload"] == {"backtest": True}
    assert build_get_response("/api/regime_ai", services, make_response=_make_response)["payload"] == {"regime": True}
    assert build_get_response("/api/recommend_portfolio", services, make_response=_make_response)["payload"] == {
        "recommend": True
    }


def test_build_get_response_unknown_route_returns_404() -> None:
    response = build_get_response("/missing", _services(), make_response=_make_response)

    assert response == {
        "kind": "not_found",
        "payload": {"error": "not found"},
        "code": 404,
        "content_type": None,
    }
