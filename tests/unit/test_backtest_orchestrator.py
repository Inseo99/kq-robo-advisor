from __future__ import annotations

import numpy as np
import pandas as pd

from kq_tool.backtest.orchestrator import run_rebalanced_strategy_backtest


def _prices(rows: int = 320) -> pd.DataFrame:
    index = pd.date_range("2023-01-02", periods=rows, freq="B")
    base = np.linspace(100, 140, rows)
    return pd.DataFrame(
        {
            "A": base * 1.20,
            "B": base * 1.05,
            "C": base[::-1] + 60,
        },
        index=index,
    )


def test_run_rebalanced_strategy_backtest_returns_legacy_payload_shape() -> None:
    price_df = _prices()
    benchmark = price_df.mean(axis=1)

    result = run_rebalanced_strategy_backtest(
        price_df,
        strategy="quant",
        top_n=2,
        rebalance="M",
        period="max",
        benchmark=benchmark,
        universe_names={"A": "Alpha", "B": "Beta", "C": "Counter"},
    )

    assert result["strategy"] == "quant"
    assert result["top_n"] == 2
    assert result["n_rebalance"] > 0
    assert len(result["equity"]) == len(result["dates"])
    assert len(result["benchmark"]) == len(result["benchmark_dates"])
    assert len(result["underwater"]) == len(result["equity"])
    assert "metrics" in result
    assert result["costs"]["transaction_cost_bps"] == 0.0
    assert result["holdings"][-1]["tickers"]


def test_run_rebalanced_strategy_backtest_applies_pit_selector() -> None:
    result = run_rebalanced_strategy_backtest(
        _prices(),
        strategy="equal",
        top_n=1,
        rebalance="Q",
        period="max",
        pit_selector=lambda _date: ["B"],
        universe_names={"B": "Beta"},
    )

    assert result["n_rebalance"] > 0
    assert all(row["tickers"] == ["Beta"] for row in result["holdings"])


def test_run_rebalanced_strategy_backtest_applies_transaction_costs() -> None:
    price_df = _prices()
    gross = run_rebalanced_strategy_backtest(
        price_df,
        strategy="quant",
        top_n=2,
        rebalance="M",
        period="max",
    )
    net = run_rebalanced_strategy_backtest(
        price_df,
        strategy="quant",
        top_n=2,
        rebalance="M",
        period="max",
        transaction_cost_bps=10,
        slippage_bps=5,
    )

    assert net["equity"][-1] < gross["equity"][-1]
    assert net["costs"]["transaction_cost_bps"] == 10.0
    assert net["costs"]["slippage_bps"] == 5.0
    assert net["costs"]["total_turnover"] > 0
