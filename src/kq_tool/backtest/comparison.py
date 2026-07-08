"""Backtest comparison response helpers."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy

from kq_tool.backtest.strategy_meta import (
    KOSPI_BENCHMARK,
    QUANT,
    QUANT_COMPARE,
    QUANT_ROBO_FILTER,
    QUANT_S2,
    QUANT_S2_ROBO_FILTER,
)


def _metric_value(result: Mapping[str, object], key: str) -> float | None:
    metrics = result.get("metrics", {})
    if not isinstance(metrics, Mapping):
        return None
    value = metrics.get(key)
    return float(value) if isinstance(value, (int, float)) else None


def _cost_value(result: Mapping[str, object], key: str) -> float | None:
    costs = result.get("costs", {})
    if not isinstance(costs, Mapping):
        return None
    value = costs.get(key)
    return float(value) if isinstance(value, (int, float)) else None


def _delta(on: float | None, off: float | None) -> float | None:
    if on is None or off is None:
        return None
    return round(on - off, 3)


def _overlay_delta_report(
    label: str,
    off_result: Mapping[str, object],
    on_result: Mapping[str, object],
) -> dict:
    report = {
        "label": label,
        "off_label": "OFF",
        "on_label": "ON",
        "rows": [
            {
                "metric": "누적수익률",
                "unit": "%p",
                "off": _metric_value(off_result, "total_return"),
                "on": _metric_value(on_result, "total_return"),
                "delta": _delta(_metric_value(on_result, "total_return"), _metric_value(off_result, "total_return")),
            },
            {
                "metric": "CAGR",
                "unit": "%p",
                "off": _metric_value(off_result, "cagr"),
                "on": _metric_value(on_result, "cagr"),
                "delta": _delta(_metric_value(on_result, "cagr"), _metric_value(off_result, "cagr")),
            },
            {
                "metric": "MDD",
                "unit": "%p",
                "off": _metric_value(off_result, "mdd"),
                "on": _metric_value(on_result, "mdd"),
                "delta": _delta(_metric_value(on_result, "mdd"), _metric_value(off_result, "mdd")),
            },
            {
                "metric": "Sharpe",
                "unit": "",
                "off": _metric_value(off_result, "sharpe"),
                "on": _metric_value(on_result, "sharpe"),
                "delta": _delta(_metric_value(on_result, "sharpe"), _metric_value(off_result, "sharpe")),
            },
            {
                "metric": "총 회전율",
                "unit": "x",
                "off": _cost_value(off_result, "total_turnover"),
                "on": _cost_value(on_result, "total_turnover"),
                "delta": _delta(_cost_value(on_result, "total_turnover"), _cost_value(off_result, "total_turnover")),
            },
        ],
    }
    audit = on_result.get("filter_audit")
    if isinstance(audit, Mapping):
        report["filter_audit"] = deepcopy(dict(audit))
    return report


def build_quant_comparison_response(
    momentum_result: Mapping[str, object],
    momentum_overlay_result: Mapping[str, object] | None,
    s2_result: Mapping[str, object],
    s2_overlay_result: Mapping[str, object] | None = None,
) -> dict:
    """Build the strategy-tab response for quant OFF/ON, S2 OFF/ON, and KOSPI."""

    payload = deepcopy(dict(momentum_result))
    payload["strategy"] = QUANT_COMPARE.key
    runs = [
        {
            "label": f"{QUANT.label} OFF",
            "color": QUANT.color,
            "data": deepcopy(dict(momentum_result)),
        }
    ]
    if momentum_overlay_result is not None:
        runs.append(
            {
                "label": f"{QUANT.label} ON",
                "color": QUANT_ROBO_FILTER.color,
                "data": deepcopy(dict(momentum_overlay_result)),
            }
        )
    runs.append(
        {
            "label": f"{QUANT_S2.label} OFF",
            "color": QUANT_S2.color,
            "data": deepcopy(dict(s2_result)),
        }
    )
    if s2_overlay_result is not None:
        runs.append(
            {
                "label": f"{QUANT_S2.label} ON",
                "color": QUANT_S2_ROBO_FILTER.color,
                "data": deepcopy(dict(s2_overlay_result)),
            }
        )
    payload["comparison_runs"] = runs
    overlay_reports = []
    if momentum_overlay_result is not None:
        overlay_reports.append(_overlay_delta_report(QUANT.label, momentum_result, momentum_overlay_result))
    if s2_overlay_result is not None:
        overlay_reports.append(_overlay_delta_report(QUANT_S2.label, s2_result, s2_overlay_result))
    payload["overlay_report"] = {
        "mode": "filter",
        "description": "로보신호는 단독 전략이 아니라 기존 퀀트 후보에서 약세 종목을 제외하고 차순위 후보로 보충하는 필터로 검증합니다.",
        "reports": overlay_reports,
    }
    payload["comparison_benchmark"] = {
        "label": KOSPI_BENCHMARK.label,
        "color": KOSPI_BENCHMARK.color,
        "dates": list(momentum_result.get("benchmark_dates", []) or []),
        "equity": list(momentum_result.get("benchmark", []) or []),
        "metrics": deepcopy(dict(momentum_result.get("bench_metrics", {}) or {})),
    }
    return payload


_build_quant_comparison_response = build_quant_comparison_response
