from __future__ import annotations

from kq_tool.api.dispatcher import (
    ApiResponse,
    apply_api_response,
    dispatch_get,
    handle_dispatched_get,
    handle_dispatched_get_safely,
    handle_legacy_get,
)
from kq_tool.backtest.strategy_meta import QUANT_COMPARE


def _services():
    return {
        "macro": lambda: {"macro": True},
        "health": lambda: {"ok": True, "health": True},
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
        "recommend_portfolio": lambda *a: {"recommend": True},
        "market_report": lambda: {"market_report": True},
    }


def test_dispatch_root_returns_index_file() -> None:
    response = dispatch_get("/", _services())

    assert response.kind == "file"
    assert response.payload == "index.html"
    assert response.content_type == "text/html; charset=utf-8"


def test_dispatch_stock_parses_query_defaults_and_overrides() -> None:
    response = dispatch_get("/api/stock?t=000660.KS&p=max", _services())

    assert response.kind == "json"
    assert response.payload == {"ticker": "000660.KS", "period": "max"}


def test_dispatch_health_route() -> None:
    response = dispatch_get("/api/health", _services())

    assert response.kind == "json"
    assert response.payload["ok"] is True
    assert response.payload["health"] is True


def test_dispatch_strategy_backtest_parses_cost_types() -> None:
    response = dispatch_get("/api/stratbt?s=robo&n=7&r=Q&p=5y&tc=10&slip=5", _services())

    assert response.payload == {
        "strategy": "robo",
        "top_n": 7,
        "rebalance": "Q",
        "period": "5y",
        "transaction_cost_bps": 10.0,
        "slippage_bps": 5.0,
    }


def test_dispatch_strategy_backtest_accepts_quant_compare() -> None:
    response = dispatch_get("/api/stratbt?s=quant_compare&n=3&r=Q&p=12y&tc=10&slip=5", _services())

    assert response.payload == {
        "strategy": QUANT_COMPARE.key,
        "top_n": 3,
        "rebalance": "Q",
        "period": "12y",
        "transaction_cost_bps": 10.0,
        "slippage_bps": 5.0,
    }




def test_dispatch_market_report_route() -> None:
    response = dispatch_get("/api/market_report", _services())

    assert response.kind == "json"
    assert response.payload == {"market_report": True}
def test_dispatch_unknown_route_returns_not_found() -> None:
    response = dispatch_get("/missing", _services())

    assert response.kind == "not_found"
    assert response.code == 404


def test_apply_api_response_routes_file_response() -> None:
    calls: list[tuple[str, object, object]] = []

    result = apply_api_response(
        ApiResponse("file", "index.html", content_type="text/html"),
        send_file=lambda payload, content_type: calls.append(("file", payload, content_type)),
        send_json=lambda payload, code: calls.append(("json", payload, code)),
        send_empty=lambda code: calls.append(("empty", code, None)),
    )

    assert result == "file"
    assert calls == [("file", "index.html", "text/html")]


def test_apply_api_response_routes_json_response() -> None:
    calls: list[tuple[str, object, object]] = []

    result = apply_api_response(
        ApiResponse("json", {"ok": True}, code=201),
        send_file=lambda payload, content_type: calls.append(("file", payload, content_type)),
        send_json=lambda payload, code: calls.append(("json", payload, code)),
        send_empty=lambda code: calls.append(("empty", code, None)),
    )

    assert result == "json"
    assert calls == [("json", {"ok": True}, 201)]


def test_apply_api_response_routes_empty_response_for_unknown_kind() -> None:
    calls: list[tuple[str, object, object]] = []

    result = apply_api_response(
        ApiResponse("not_found", {"error": "not found"}, code=404),
        send_file=lambda payload, content_type: calls.append(("file", payload, content_type)),
        send_json=lambda payload, code: calls.append(("json", payload, code)),
        send_empty=lambda code: calls.append(("empty", code, None)),
    )

    assert result == "not_found"
    assert calls == [("empty", 404, None)]


def test_handle_dispatched_get_dispatches_and_applies_file_json_and_empty() -> None:
    calls: list[tuple[str, object, object]] = []

    common_callbacks = {
        "send_file": lambda payload, content_type: calls.append(("file", payload, content_type)),
        "send_json": lambda payload, code: calls.append(("json", payload, code)),
        "send_empty": lambda code: calls.append(("empty", code, None)),
    }

    assert handle_dispatched_get("/", _services(), **common_callbacks) == "file"
    assert handle_dispatched_get("/api/ping", _services(), **common_callbacks) == "json"
    assert handle_dispatched_get("/missing", _services(), **common_callbacks) == "not_found"

    assert calls == [
        ("file", "index.html", "text/html; charset=utf-8"),
        ("json", {"ok": True}, 200),
        ("empty", 404, None),
    ]


def test_handle_dispatched_get_safely_returns_success_result() -> None:
    calls: list[tuple[str, object, object]] = []

    result = handle_dispatched_get_safely(
        "/api/ping",
        _services(),
        send_file=lambda payload, content_type: calls.append(("file", payload, content_type)),
        send_json=lambda payload, code: calls.append(("json", payload, code)),
        send_empty=lambda code: calls.append(("empty", code, None)),
        send_error=lambda exc: calls.append(("error", str(exc), None)),
    )

    assert result == {"ok": True, "kind": "json"}
    assert calls == [("json", {"ok": True}, 200)]


def test_handle_dispatched_get_safely_converts_exception_to_error_callback() -> None:
    calls: list[tuple[str, object, object]] = []
    errors: list[str] = []
    services = _services()
    services["health"] = lambda: (_ for _ in ()).throw(RuntimeError("health down"))

    result = handle_dispatched_get_safely(
        "/api/health",
        services,
        send_file=lambda payload, content_type: calls.append(("file", payload, content_type)),
        send_json=lambda payload, code: calls.append(("json", payload, code)),
        send_empty=lambda code: calls.append(("empty", code, None)),
        send_error=lambda exc: calls.append(("error", str(exc), None)),
        on_error=lambda exc: errors.append(str(exc)),
    )

    assert result["ok"] is False
    assert isinstance(result["error"], RuntimeError)
    assert calls == [("error", "health down", None)]
    assert errors == ["health down"]


def test_handle_legacy_get_routes_file_json_actions_and_not_found() -> None:
    calls: list[tuple[str, object, object]] = []

    common_callbacks = {
        "send_file": lambda payload, content_type: calls.append(("file", payload, content_type)),
        "send_json": lambda payload, code: calls.append(("json", payload, code)),
        "send_empty": lambda code: calls.append(("empty", code, None)),
        "stock": lambda query: calls.append(("stock", query["t"][0], None)),
        "screen": lambda: calls.append(("screen", None, None)),
        "backtest": lambda: calls.append(("backtest", None, None)),
        "stratbt": lambda query: calls.append(("stratbt", query["s"][0], None)),
        "macro_payload": {"macro": True},
        "regime_ai": lambda: calls.append(("regime_ai", None, None)),
        "recommend_portfolio": lambda *a: calls.append(("recommend", None, None)),
        "market_report": lambda: calls.append(("market_report", None, None)),
        "health": lambda: {"health": True},
    }

    assert handle_legacy_get("/", **common_callbacks) == "file"
    assert handle_legacy_get("/api/stock?t=005930.KS", **common_callbacks) == "stock"
    assert handle_legacy_get("/api/stratbt?s=quant", **common_callbacks) == "stratbt"
    assert handle_legacy_get("/api/macro", **common_callbacks) == "macro"
    assert handle_legacy_get("/api/health", **common_callbacks) == "health"
    assert handle_legacy_get("/api/ping", **common_callbacks) == "ping"
    assert handle_legacy_get("/missing", **common_callbacks) == "not_found"

    assert calls == [
        ("file", "index.html", "text/html; charset=utf-8"),
        ("stock", "005930.KS", None),
        ("stratbt", "quant", None),
        ("json", {"macro": True}, 200),
        ("json", {"health": True}, 200),
        ("json", {"ok": True}, 200),
        ("empty", 404, None),
    ]


def test_handle_legacy_get_routes_zero_arg_endpoint_actions() -> None:
    calls: list[str] = []
    callbacks = {
        "send_file": lambda payload, content_type: None,
        "send_json": lambda payload, code: None,
        "send_empty": lambda code: None,
        "stock": lambda query: None,
        "screen": lambda: calls.append("screen"),
        "backtest": lambda: calls.append("backtest"),
        "stratbt": lambda query: None,
        "macro_payload": {},
        "regime_ai": lambda: calls.append("regime_ai"),
        "recommend_portfolio": lambda *a: calls.append("recommend_portfolio"),
        "market_report": lambda: calls.append("market_report"),
        "health": lambda: {},
    }

    assert handle_legacy_get("/api/screen", **callbacks) == "screen"
    assert handle_legacy_get("/api/backtest", **callbacks) == "backtest"
    assert handle_legacy_get("/api/regime_ai", **callbacks) == "regime_ai"
    assert handle_legacy_get("/api/recommend_portfolio", **callbacks) == "recommend_portfolio"
    assert handle_legacy_get("/api/market_report", **callbacks) == "market_report"
    assert calls == ["screen", "backtest", "regime_ai", "recommend_portfolio", "market_report"]
def test_server_uses_safe_and_legacy_dispatch_helpers() -> None:
    import server

    assert server._kq_handle_dispatched_get_safely is handle_dispatched_get_safely
    assert server._kq_handle_legacy_get is handle_legacy_get
    assert not hasattr(server, "_kq_apply_api_response")
    assert not hasattr(server, "_kq_handle_dispatched_get")



