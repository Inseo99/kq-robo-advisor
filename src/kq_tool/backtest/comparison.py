"""Backtest comparison response helpers."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy

from kq_tool.backtest.strategy_meta import KOSPI_BENCHMARK, QUANT, QUANT_COMPARE, QUANT_S2, ROBO


def build_quant_comparison_response(
    momentum_result: Mapping[str, object],
    s2_result: Mapping[str, object],
    robo_result: Mapping[str, object] | None = None,
) -> dict:
    """Build the strategy-tab response for momentum, S2, optional robo, and KOSPI."""

    payload = deepcopy(dict(momentum_result))
    payload["strategy"] = QUANT_COMPARE.key
    runs = [
        {
            "label": QUANT.label,
            "color": QUANT.color,
            "data": deepcopy(dict(momentum_result)),
        },
        {
            "label": QUANT_S2.label,
            "color": QUANT_S2.color,
            "data": deepcopy(dict(s2_result)),
        },
    ]
    if robo_result is not None:
        runs.append(
            {
                "label": ROBO.label,
                "color": ROBO.color,
                "data": deepcopy(dict(robo_result)),
            }
        )
    payload["comparison_runs"] = runs
    payload["comparison_benchmark"] = {
        "label": KOSPI_BENCHMARK.label,
        "color": KOSPI_BENCHMARK.color,
        "dates": list(momentum_result.get("benchmark_dates", []) or []),
        "equity": list(momentum_result.get("benchmark", []) or []),
        "metrics": deepcopy(dict(momentum_result.get("bench_metrics", {}) or {})),
    }
    return payload


_build_quant_comparison_response = build_quant_comparison_response
