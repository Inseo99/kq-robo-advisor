from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_smoke_api():
    path = Path(__file__).resolve().parents[1] / "smoke_api.py"
    spec = importlib.util.spec_from_file_location("kq_smoke_api", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_strategy_backtest_check_validates_expected_strategy_and_benchmark() -> None:
    smoke = _load_smoke_api()

    name, ok, payload = smoke.strategy_backtest_check(
        "stratbt_s2",
        {
            "strategy": "quant_s2",
            "metrics": {"cagr": 1},
            "benchmark": [100, 101],
            "costs": {"total": 1},
            "n_rebalance": 3,
        },
        expected_strategy="quant_s2",
        require_benchmark=True,
    )

    assert name == "stratbt_s2"
    assert ok is True
    assert payload == {
        "strategy": "quant_s2",
        "n_rebalance": 3,
        "metrics": True,
        "benchmark": True,
        "costs": True,
    }


def test_strategy_backtest_check_fails_on_wrong_strategy() -> None:
    smoke = _load_smoke_api()

    _, ok, _ = smoke.strategy_backtest_check(
        "stratbt_s2",
        {"strategy": "quant", "metrics": {"cagr": 1}, "benchmark": [100]},
        expected_strategy="quant_s2",
        require_benchmark=True,
    )

    assert ok is False


def test_quant_compare_check_requires_two_runs_and_benchmark() -> None:
    smoke = _load_smoke_api()

    name, ok, payload = smoke.quant_compare_check(
        {
            "strategy": "quant_compare",
            "comparison_runs": [
                {"label": "퀀트(모멘텀) OFF"},
                {"label": "퀀트(모멘텀) ON"},
                {"label": "퀀트(S2모멘텀) OFF"},
                {"label": "퀀트(S2모멘텀) ON"},
            ],
            "overlay_report": {"reports": [{"label": "퀀트(모멘텀)"}]},
            "comparison_benchmark": {"equity": [100, 101]},
        }
    )

    assert name == "stratbt_compare"
    assert ok is True
    assert payload == {
        "runs": [
            "퀀트(모멘텀) OFF",
            "퀀트(모멘텀) ON",
            "퀀트(S2모멘텀) OFF",
            "퀀트(S2모멘텀) ON",
        ],
        "overlay_report": True,
        "benchmark": True,
    }


def test_quant_compare_check_fails_without_benchmark_equity() -> None:
    smoke = _load_smoke_api()

    _, ok, _ = smoke.quant_compare_check(
        {
            "strategy": "quant_compare",
            "comparison_runs": [{"label": "A"}, {"label": "B"}],
            "comparison_benchmark": {},
        }
    )

    assert ok is False
