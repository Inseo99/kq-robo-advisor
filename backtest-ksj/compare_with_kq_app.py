"""KQ 앱 전략검증 탭과 backtest-ksj 결과 비교표 생성.

목적
----
두 결과를 하나의 성과표로 "직접 우열 비교"하려는 스크립트가 아니다.
앱 전략검증 탭과 backtest-ksj는 유니버스, 종목 수, 비용, 전략 정의가 다르므로
발표에서는 서로 다른 검증 축으로 설명해야 한다.

이 스크립트는 그 차이를 명시한 뒤, 각 엔진의 대표 결과를 한 표에 정리한다.

실행
----
    cd backtest-ksj
    python compare_with_kq_app.py

출력
----
    results/kq_app_vs_backtest_ksj.csv
    results/kq_app_vs_backtest_ksj.md
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd

import config as C


HERE = Path(__file__).resolve().parent
REPO = HERE.parent
RESULTS = HERE / "results"


APP_DEFAULTS = {
    "strategy": "quant_compare",
    "top_n": 5,
    "rebalance": "M",
    "period": "12y",
    "transaction_cost_bps": 10,
    "slippage_bps": 5,
}


KSJ_STRATEGY_LABELS = {
    "s1m": "S1 월간 모멘텀20",
    "s2m": "S2 월간 모멘텀20 + t1 위험회피",
    "s3m": "S3 월간 모멘텀20 + t2 추세추종",
    "s1w": "S1 주간 모멘텀20",
    "s2w": "S2 주간 모멘텀20 + t1 위험회피",
    "s3w": "S3 주간 모멘텀20 + t2 추세추종",
    "s1m-kd200": "S1 월간 KODEX200",
    "s2m-kd200": "S2 월간 KODEX200 + t1 위험회피",
    "s3m-kd200": "S3 월간 KODEX200 + t2 추세추종",
    "s1w-kd200": "S1 주간 KODEX200",
    "s2w-kd200": "S2 주간 KODEX200 + t1 위험회피",
    "s3w-kd200": "S3 주간 KODEX200 + t2 추세추종",
    "069500": "KODEX200 벤치마크",
}


def _pct_from_decimal(x: Any) -> float | None:
    if pd.isna(x):
        return None
    return round(float(x) * 100, 2)


def _num(x: Any, digits: int = 3) -> float | None:
    if pd.isna(x):
        return None
    return round(float(x), digits)


def _metric(metrics: dict[str, Any], key: str) -> float | None:
    val = metrics.get(key)
    if isinstance(val, (int, float)):
        return round(float(val), 3 if key in {"sharpe", "calmar"} else 2)
    return None


def _app_total_cost_pct(costs: dict[str, Any]) -> float | None:
    """App payload stores cumulative costs as a rate, e.g. 0.2553 = 25.53%."""

    val = costs.get("total_cost_rate")
    if isinstance(val, (int, float)):
        return round(float(val) * 100, 3)
    val = costs.get("total_cost_pct")
    if isinstance(val, (int, float)):
        return round(float(val), 3)
    return None


def _base_app_label(label: str) -> str:
    """Normalize KQ app OFF/ON row labels to the overlay-report base label."""

    return label.replace(" ON", "").replace(" OFF", "").strip()


def load_backtest_ksj_rows() -> list[dict[str, Any]]:
    path = RESULTS / "summary_full_oos.csv"
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)
    rows: list[dict[str, Any]] = []
    for _, r in df.iterrows():
        code = str(r["code"])
        cost_total_pct = _num(r["cost_total_pct"])
        turnover = None
        if cost_total_pct is not None and C.COST_ONE_WAY > 0:
            turnover = round(cost_total_pct / (C.COST_ONE_WAY * 100), 3)
        rows.append(
            {
                "engine": "backtest-ksj",
                "strategy": KSJ_STRATEGY_LABELS.get(code, code),
                "code": code,
                "role": "별도 검증 엔진",
                "universe": "PiT 시총 상위 300 (KOSPI+KOSDAQ)",
                "selection": "모멘텀20 또는 KODEX200 단일",
                "rebalance": "월간" if r["cadence"] == "M" else "주간",
                "period": f"{r['start']} ~ {r['end']}",
                "cost_assumption": "왕복 0.5%, 현금 연 2%",
                "return_basis": "net, 가격수익률",
                "total_return_pct": _pct_from_decimal(r["total_return"]),
                "cagr_pct": _pct_from_decimal(r["cagr"]),
                "mdd_pct": _pct_from_decimal(r["mdd"]),
                "sharpe": _num(r["sharpe"]),
                "calmar": _num(r["calmar"]),
                "trades": int(r["trades"]) if pd.notna(r["trades"]) else None,
                "total_turnover_x": turnover,
                "cost_total_pct": cost_total_pct,
                "n_rebalance": int(r["n_rebal_invested"]) if pd.notna(r["n_rebal_invested"]) else None,
                "note": "Fold4/전체 OOS는 2025~2026 합성 가격 영향으로 절대 CAGR 해석 금지",
            }
        )
    return rows


def load_kq_app_rows() -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    sys.path.insert(0, str(REPO))
    import server  # noqa: WPS433 - local app integration point

    payload = server.run_strategy_backtest(
        APP_DEFAULTS["strategy"],
        APP_DEFAULTS["top_n"],
        APP_DEFAULTS["rebalance"],
        APP_DEFAULTS["period"],
        APP_DEFAULTS["transaction_cost_bps"],
        APP_DEFAULTS["slippage_bps"],
    )
    if payload.get("error"):
        raise RuntimeError(payload["error"])

    rows: list[dict[str, Any]] = []
    audit_by_label: dict[str, dict[str, Any]] = {}
    overlay_report = payload.get("overlay_report")
    if isinstance(overlay_report, dict):
        for rep in overlay_report.get("reports", []) or []:
            if not isinstance(rep, dict):
                continue
            audit = rep.get("filter_audit")
            label = rep.get("label")
            if isinstance(label, str) and isinstance(audit, dict):
                audit_by_label[label] = audit

    runs = payload.get("comparison_runs") or []
    for item in runs:
        label = item.get("label", "")
        data = item.get("data", {}) or {}
        metrics = data.get("metrics", {}) or {}
        costs = data.get("costs", {}) or {}
        audit = audit_by_label.get(_base_app_label(label), {}) if label.endswith(" ON") else {}
        rows.append(
            {
                "engine": "KQ app",
                "strategy": label,
                "code": data.get("strategy"),
                "role": "실시간 앱 전략검증 탭",
                "universe": "앱 수정주가 패널 + PiT 시총 필터",
                "selection": "top 5, 로보필터 OFF/ON",
                "rebalance": "월간",
                "period": "2014~현재(앱 12y 고정 시작 정책)",
                "cost_assumption": "거래비용 10bps + 슬리피지 5bps",
                "return_basis": data.get("returns_basis", "net"),
                "total_return_pct": _metric(metrics, "total_return"),
                "cagr_pct": _metric(metrics, "cagr"),
                "mdd_pct": _metric(metrics, "mdd"),
                "sharpe": _metric(metrics, "sharpe"),
                "calmar": _metric(metrics, "calmar"),
                "trades": None,
                "total_turnover_x": _metric(costs, "total_turnover"),
                "cost_total_pct": _app_total_cost_pct(costs),
                "n_rebalance": data.get("n_rebalance"),
                "filter_events": audit.get("n_events"),
                "filter_hit_count": audit.get("hit_count"),
                "filter_hit_n": audit.get("hit_n"),
                "filter_spread_pct": audit.get("avg_spread"),
                "filter_spread_n": audit.get("avg_spread_n"),
                "filter_spread_t_stat": audit.get("avg_spread_t_stat"),
                "filter_spread_p_value": audit.get("avg_spread_p_value"),
                "filter_positive_periods": audit.get("positive_periods"),
                "filter_periods_tested": audit.get("periods_tested"),
                "filter_hit_rate_pct": audit.get("hit_rate"),
                "filter_hit_p_value": audit.get("hit_p_value"),
                "note": "앱 화면 기본 조건. backtest-ksj와 유니버스/종목수/비용 정의가 다름",
            }
        )

    bench = payload.get("comparison_benchmark", {}) or {}
    bench_metrics = bench.get("metrics", {}) or {}
    if bench_metrics:
        rows.append(
            {
                "engine": "KQ app",
                "strategy": bench.get("label", "KOSPI"),
                "code": "KOSPI",
                "role": "앱 비교 벤치마크",
                "universe": "KOSPI",
                "selection": "지수",
                "rebalance": "-",
                "period": "앱 전략검증과 동일",
                "cost_assumption": "비용 없음",
                "return_basis": "price index",
                "total_return_pct": _metric(bench_metrics, "total_return"),
                "cagr_pct": _metric(bench_metrics, "cagr"),
                "mdd_pct": _metric(bench_metrics, "mdd"),
                "sharpe": _metric(bench_metrics, "sharpe"),
                "calmar": _metric(bench_metrics, "calmar"),
                "trades": 0,
                "total_turnover_x": 0.0,
                "cost_total_pct": 0.0,
                "n_rebalance": 0,
                "filter_events": None,
                "filter_hit_count": None,
                "filter_hit_n": None,
                "filter_spread_pct": None,
                "filter_spread_n": None,
                "filter_spread_t_stat": None,
                "filter_spread_p_value": None,
                "filter_positive_periods": None,
                "filter_periods_tested": None,
                "filter_hit_rate_pct": None,
                "filter_hit_p_value": None,
                "note": "앱 벤치마크",
            }
        )
    return rows, overlay_report


def write_markdown(df: pd.DataFrame, overlay_report: dict[str, Any] | None) -> None:
    out = RESULTS / "kq_app_vs_backtest_ksj.md"
    view_cols = [
        "engine",
        "strategy",
        "rebalance",
        "total_return_pct",
        "cagr_pct",
        "mdd_pct",
        "sharpe",
        "calmar",
        "trades",
        "total_turnover_x",
        "cost_total_pct",
        "n_rebalance",
        "filter_events",
        "filter_hit_count",
        "filter_hit_n",
        "filter_spread_pct",
        "filter_spread_n",
        "filter_spread_t_stat",
        "filter_spread_p_value",
        "filter_positive_periods",
        "filter_periods_tested",
        "filter_hit_rate_pct",
        "filter_hit_p_value",
    ]
    lines: list[str] = []
    lines.append("# KQ 앱 전략검증 vs backtest-ksj 비교")
    lines.append("")
    lines.append("## 해석 원칙")
    lines.append("")
    lines.append("- 이 표는 두 엔진의 우열을 직접 판정하기 위한 표가 아니다.")
    lines.append("- KQ 앱은 실시간 의사결정 화면이며, `backtest-ksj`는 발표/검증용 독립 엔진이다.")
    lines.append("- 두 엔진은 유니버스, 종목 수, 비용, 벤치마크 기준이 다르므로 숫자는 역할별로 해석한다.")
    lines.append("")
    lines.append("## 실행 조건")
    lines.append("")
    lines.append(
        f"- KQ 앱: `{APP_DEFAULTS['strategy']}`, top_n={APP_DEFAULTS['top_n']}, "
        f"rebalance={APP_DEFAULTS['rebalance']}, period={APP_DEFAULTS['period']}, "
        f"cost={APP_DEFAULTS['transaction_cost_bps']}bps, slip={APP_DEFAULTS['slippage_bps']}bps"
    )
    lines.append("- backtest-ksj: `summary_full_oos.csv`의 전체 OOS 결과")
    lines.append("- 규약 차이: KQ 앱은 KOSPI 비교, 종가 기반 수정주가 패널, 10bps+5bps 비용, 앱 `RISK_FREE_RATE`를 사용한다.")
    lines.append("- 규약 차이: backtest-ksj는 KODEX200 벤치마크, 월말/금요일 평가 후 익영업일 시초가 리밸런싱, 왕복 0.5%, 현금 연 2%를 사용한다.")
    lines.append("")
    lines.append("## 성과 비교표")
    lines.append("")
    table = df[view_cols].copy()
    table = table.where(pd.notna(table), "")
    lines.append(table.to_markdown(index=False))
    if overlay_report and overlay_report.get("reports"):
        lines.append("")
        lines.append("## 앱 로보필터 ON/OFF 기여도")
        for rep in overlay_report["reports"]:
            lines.append("")
            lines.append(f"### {rep.get('label')}")
            audit = rep.get("filter_audit") or {}
            if audit:
                lines.append(
                    "- 제외 감사: "
                    f"이벤트 {audit.get('n_events')}회, "
                    f"제외 {audit.get('n_excluded')}개, "
                    f"대체 {audit.get('n_replacements')}개, "
                    f"대체-제외 {audit.get('avg_spread')}%p, "
                    f"격차 t={audit.get('avg_spread_t_stat')}, p={audit.get('avg_spread_p_value')}, "
                    f"구간+ {audit.get('positive_periods')}/{audit.get('periods_tested')}, "
                    f"Hit {audit.get('hit_count')}/{audit.get('hit_n')} "
                    f"({audit.get('hit_rate')}%, p={audit.get('hit_p_value')})"
                )
            r_df = pd.DataFrame(rep.get("rows", []))
            if not r_df.empty:
                lines.append(r_df.to_markdown(index=False))
    lines.append("")
    lines.append("## 발표 문구")
    lines.append("")
    lines.append(
        "> 앱 전략검증은 실제 서비스 화면의 의사결정 보조 성능을 보여주고, "
        "`backtest-ksj`는 별도 검증 엔진으로 사전 고정 모멘텀/위험회피 규칙의 "
        "구간별 강건성을 점검한다. 두 결과는 같은 결론을 강요하지 않고, "
        "서로 다른 검증 축으로 해석한다."
    )
    lines.append("")
    lines.append(
        "> 로보필터는 전 전략 일괄 적용이 아니라 제외 감사 지표로 적용 범위를 결정한다. "
        "기여 증거가 확인될 때만 적용 후보로 두고, 증거가 없거나 평균 격차 방향이 음수면 기본값인 미적용을 유지한다."
    )
    out.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    os.makedirs(RESULTS, exist_ok=True)
    app_rows, overlay_report = load_kq_app_rows()
    ksj_rows = load_backtest_ksj_rows()
    df = pd.DataFrame(app_rows + ksj_rows)
    out_csv = RESULTS / "kq_app_vs_backtest_ksj.csv"
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    write_markdown(df, overlay_report)
    print(f"[done] {out_csv}")
    print(f"[done] {RESULTS / 'kq_app_vs_backtest_ksj.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
