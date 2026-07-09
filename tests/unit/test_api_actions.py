from __future__ import annotations

from kq_tool.api.actions import run_json_service_action, run_stock_action, run_strategy_backtest_action


def test_run_json_service_action_delegates_zero_arg_service_to_json_action() -> None:
    calls: list[tuple[str, object]] = []

    def service() -> dict[str, bool]:
        calls.append(("service", None))
        return {"ok": True}

    def json_action(action):
        calls.append(("json_action", None))
        return action()

    result = run_json_service_action(service, json_action=json_action)

    assert result == {"ok": True}
    assert calls == [("json_action", None), ("service", None)]

def test_run_stock_action_parses_query_and_delegates_to_json_action() -> None:
    calls: list[tuple[str, object]] = []

    def analyze_stock(ticker: str, period: str) -> dict[str, str]:
        calls.append(("analyze", (ticker, period)))
        return {"ticker": ticker, "period": period}

    def json_action(action):
        calls.append(("json_action", None))
        return action()

    result = run_stock_action(
        {"t": ["000660.KS"], "p": ["max"]},
        analyze_stock=analyze_stock,
        json_action=json_action,
    )

    assert result == {"ticker": "000660.KS", "period": "max"}
    assert calls == [("json_action", None), ("analyze", ("000660.KS", "max"))]


def test_run_stock_action_preserves_legacy_defaults() -> None:
    result = run_stock_action(
        {},
        analyze_stock=lambda ticker, period: {"ticker": ticker, "period": period},
        json_action=lambda action: action(),
    )

    assert result == {"ticker": "005930.KS", "period": "1y"}


def test_run_strategy_backtest_action_parses_query_and_delegates_to_json_action() -> None:
    calls: list[tuple[str, object]] = []

    def run_strategy_backtest(strategy, top_n, rebalance, period, transaction_cost_bps, slippage_bps):
        calls.append(
            (
                "stratbt",
                (strategy, top_n, rebalance, period, transaction_cost_bps, slippage_bps),
            )
        )
        return {"strategy": strategy, "top_n": top_n, "tc": transaction_cost_bps, "slip": slippage_bps}

    def json_action(action):
        calls.append(("json_action", None))
        return action()

    result = run_strategy_backtest_action(
        {"s": ["robo"], "n": ["7"], "r": ["Q"], "p": ["5y"], "tc": ["10"], "slip": ["5"]},
        run_strategy_backtest=run_strategy_backtest,
        json_action=json_action,
    )

    assert result == {"strategy": "robo", "top_n": 7, "tc": 10.0, "slip": 5.0}
    assert calls == [("json_action", None), ("stratbt", ("robo", 7, "Q", "5y", 10.0, 5.0))]


def test_run_strategy_backtest_action_uses_standard_trading_cost_defaults() -> None:
    result = run_strategy_backtest_action(
        {},
        run_strategy_backtest=lambda strategy, top_n, rebalance, period, transaction_cost_bps, slippage_bps: {
            "strategy": strategy,
            "top_n": top_n,
            "rebalance": rebalance,
            "period": period,
            "tc": transaction_cost_bps,
            "slip": slippage_bps,
        },
        json_action=lambda action: action(),
    )

    assert result == {
        "strategy": "quant",
        "top_n": 5,
        "rebalance": "M",
        "period": "3y",
        "tc": 10.0,
        "slip": 5.0,
    }


def test_server_reuses_api_action_helpers() -> None:
    import server

    assert server._kq_run_json_service_action is run_json_service_action
    assert server._kq_run_stock_action is run_stock_action
    assert server._kq_run_strategy_backtest_action is run_strategy_backtest_action

