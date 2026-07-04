"""Backtest comparison response helpers."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy

from kq_tool.backtest.strategy_meta import KOSPI_BENCHMARK, QUANT, QUANT_COMPARE, QUANT_S2


def build_quant_comparison_response(
    momentum_result: Mapping[str, object],
    s2_result: Mapping[str, object],
) -> dict:
    """Build the strategy-tab response for quant momentum vs S2 momentum vs KOSPI."""

    payload = deepcopy(dict(momentum_result))
    payload["strategy"] = QUANT_COMPARE.key
    payload["comparison_runs"] = [
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
    payload["comparison_benchmark"] = {
        "label": KOSPI_BENCHMARK.label,
        "color": KOSPI_BENCHMARK.color,
        "dates": list(momentum_result.get("benchmark_dates", []) or []),
        "equity": list(momentum_result.get("benchmark", []) or []),
        "metrics": deepcopy(dict(momentum_result.get("bench_metrics", {}) or {})),
    }
    return payload


_build_quant_comparison_response = build_quant_comparison_response
