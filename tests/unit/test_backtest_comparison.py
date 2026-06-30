from __future__ import annotations

from kq_tool.backtest.comparison import build_quant_comparison_response
from kq_tool.backtest.strategy_meta import KOSPI_BENCHMARK, QUANT, QUANT_COMPARE, QUANT_S2


def test_build_quant_comparison_response_preserves_legacy_base_shape() -> None:
    quant = {
        "strategy": "quant",
        "equity": [100, 110],
        "dates": ["2025-01-31", "2025-02-28"],
        "benchmark": [100, 105],
        "benchmark_dates": ["2025-01-31", "2025-02-28"],
        "bench_metrics": {"cagr": 5.0},
        "metrics": {"cagr": 10.0},
    }
    s2 = {
        "strategy": QUANT_S2.key,
        "equity": [100, 120],
        "dates": ["2025-01-31", "2025-02-28"],
        "metrics": {"cagr": 20.0},
    }

    result = build_quant_comparison_response(quant, s2)

    assert result["strategy"] == QUANT_COMPARE.key
    assert result["metrics"] == {"cagr": 10.0}
    assert [run["label"] for run in result["comparison_runs"]] == [
        QUANT.label,
        QUANT_S2.label,
    ]
    assert result["comparison_runs"][0]["data"]["strategy"] == QUANT.key
    assert result["comparison_runs"][1]["data"]["strategy"] == QUANT_S2.key
    assert result["comparison_benchmark"] == {
        "label": KOSPI_BENCHMARK.label,
        "color": KOSPI_BENCHMARK.color,
        "dates": ["2025-01-31", "2025-02-28"],
        "equity": [100, 105],
        "metrics": {"cagr": 5.0},
    }


def test_build_quant_comparison_response_does_not_mutate_inputs() -> None:
    quant = {"strategy": QUANT.key, "benchmark": [100], "benchmark_dates": ["2025-01-31"]}
    s2 = {"strategy": QUANT_S2.key, "equity": [100]}

    result = build_quant_comparison_response(quant, s2)
    result["comparison_runs"][0]["data"]["strategy"] = "changed"
    result["comparison_benchmark"]["equity"].append(101)

    assert quant["strategy"] == QUANT.key
    assert quant["benchmark"] == [100]


def test_server_reuses_quant_comparison_helper() -> None:
    import server

    assert server._kq_build_quant_comparison_response is build_quant_comparison_response


def test_server_quant_compare_calls_both_quant_strategies(monkeypatch) -> None:
    import server

    calls: list[tuple[str, int, str, str, float, float]] = []

    def fake_run_strategy_backtest(strategy, top_n, rebalance, period, transaction_cost_bps=0.0, slippage_bps=0.0):
        calls.append((strategy, top_n, rebalance, period, transaction_cost_bps, slippage_bps))
        return {
            "strategy": strategy,
            "metrics": {"cagr": 1.0 if strategy == QUANT.key else 2.0},
            "benchmark": [100, 101],
            "benchmark_dates": ["2025-01-31", "2025-02-28"],
            "bench_metrics": {"cagr": 0.5},
        }

    monkeypatch.setattr(server, "run_strategy_backtest", fake_run_strategy_backtest)

    result = server._run_quant_comparison_backtest(7, "Q", "5y", 10.0, 5.0)

    assert calls == [
        (QUANT.key, 7, "Q", "5y", 10.0, 5.0),
        (QUANT_S2.key, 7, "Q", "5y", 10.0, 5.0),
    ]
    assert result["strategy"] == QUANT_COMPARE.key
    assert [run["data"]["strategy"] for run in result["comparison_runs"]] == [QUANT.key, QUANT_S2.key]


def test_server_quant_compare_returns_contextual_error(monkeypatch) -> None:
    import server

    def fake_run_strategy_backtest(strategy, *args, **kwargs):
        if strategy == QUANT_S2.key:
            return {"error": "리밸런싱 기간 부족"}
        return {"strategy": strategy, "metrics": {}}

    monkeypatch.setattr(server, "run_strategy_backtest", fake_run_strategy_backtest)

    result = server._run_quant_comparison_backtest(3, "Q", "1y", 0.0, 0.0)

    assert result == {"error": f"{QUANT_S2.label}: 리밸런싱 기간 부족"}


def test_run_strategy_backtest_dispatches_quant_compare_to_dedicated_cache(monkeypatch) -> None:
    import server

    captured = {}

    def fake_cached(key, ttl, fn, *args):
        captured["key"] = key
        captured["ttl"] = ttl
        captured["fn"] = fn
        captured["args"] = args
        return {"ok": True}

    monkeypatch.setattr(server, "_cached", fake_cached)

    result = server.run_strategy_backtest(QUANT_COMPARE.key, 4, "Q", "5y", 10, 5)

    assert result == {"ok": True}
    assert captured == {
        "key": "stratbt_compare:4:Q:5y:tc10.0:slip5.0",
        "ttl": 1800,
        "fn": server._run_quant_comparison_backtest,
        "args": (4, "Q", "5y", 10.0, 5.0),
    }
