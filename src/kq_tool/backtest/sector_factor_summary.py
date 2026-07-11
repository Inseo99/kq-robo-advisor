# -*- coding: utf-8 -*-
"""섹터·팩터 독립 백테스트(backtest-ksj 계열) 요약 — 전략검증 탭 표시용.

이 모듈은 결과를 '읽어서 보여주기만' 한다. 백테스트 실행·추천 엔진 반영과는
무관하며, 수치의 정본은 data/strategy_validation/sector_factor_backtest.csv 하나다.

표기 규칙 (팀 합의):
  - t1/t2는 '위험회피 게이트'로만 부른다 (메인 시스템의 '국면'과 다른 개념).
  - 조건부 위험회피 결합(국면 확률 활용)은 수치 미채택 — 후속 검증 후보 문구로만 표시.
"""
from __future__ import annotations

import csv
import os

from kq_tool.screener.strategies import DISPLAY_NAME_QUALITY

DATA_REL = os.path.join("data", "strategy_validation", "sector_factor_backtest.csv")
CURVE_REL = os.path.join("backtest-ksj", "results", "period_records_s1m_us_sec.csv")

DISPLAY_CODES = {
    "s1m_lsv": "M1 LSV",
    "s2m_lsv": "M2 LSV",
    "s3m_lsv": "M3 LSV",
    "s1m_kang": f"M1 {DISPLAY_NAME_QUALITY}",
    "s2m_kang": f"M2 {DISPLAY_NAME_QUALITY}",
    "s3m_kang": f"M3 {DISPLAY_NAME_QUALITY}",
    "069500": "KODEX200",
}

DISPLAY_SELECTIONS = {
    "Kang 우량주": DISPLAY_NAME_QUALITY,
    "중형성장주": DISPLAY_NAME_QUALITY,
}

POSITION = (
    "독립 백테스트 — 메인 추천 엔진과 분리된 전략검증 레이어이며 추천 엔진에는 "
    "반영되지 않습니다. 화면에는 M1/M2/M3의 6개 기본 모멘텀 전략만 게시합니다. "
    "미국 섹터 모멘텀 상위 3섹터 → 한국 섹터 매핑 → 팩터 종목선정"
    "(7/5/3 동일가중), 월간 리밸런스, 거래비용 반영."
)
GATES = (
    "t1 — 미국 신호 3종 중 2개 이상 점등 시 전액 현금 "
    "(S&P500 < 9개월 이평 · VIX > 18.6 · 신용스프레드 z > 1.78) / "
    "t2 — KODEX200 종가 < 10개월 이평 시 전액 현금."
)
CANDIDATE = (
    "조건부 위험회피 결합은 기존 실험에서 방어 가능성을 보였으나, 현재 채택 라벨과 "
    "2개월 관측 지연 기준에서는 아직 재검증 전입니다. 기존 국면 정의 또는 불완전한 "
    "라벨 기준의 수치는 게시하지 않고, 후속 검증 후보로 분리합니다."
)
PRINCIPLE = (
    "성과가 좋아 보이는 전략이라도 동일한 검증 기준을 통과하기 전까지 "
    "추천 엔진에 반영하지 않습니다."
)

_NUM_FIELDS = ("cagr_pct", "mdd_pct", "sharpe", "calmar", "trades",
               "cost_total_pct", "cost_ann_pct")


def _display_code(code: str) -> str:
    return DISPLAY_CODES.get(code, code)


def _display_selection(selection: str) -> str:
    return DISPLAY_SELECTIONS.get(selection, selection)


def _load_sample_curve(base_dir: str) -> dict | None:
    """기간별 성과를 눈으로 확인하기 위한 1개 전략 NAV 예시."""
    path = os.path.join(base_dir, CURVE_REL)
    if not os.path.exists(path):
        return None

    dates = []
    nav = []
    with open(path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            date = row.get("exit") or row.get("entry") or row.get("eval")
            equity = row.get("equity")
            if not date or equity in (None, ""):
                continue
            try:
                value = float(equity)
            except ValueError:
                continue
            dates.append(date)
            nav.append(value)

    if len(dates) < 2:
        return None

    base = nav[0] if nav[0] else 1.0
    norm_nav = [round(v / base * 100.0, 4) for v in nav]
    return {
        "title": "M1 섹터 모멘텀 기간별 NAV 예시",
        "note": (
            "전략 중 1개 경로를 예시로 표시합니다. 조건부 위험회피 결합 수치는 "
            "포함하지 않습니다."
        ),
        "source": CURVE_REL,
        "dates": dates,
        "nav": norm_nav,
    }


def build_sector_factor_summary(base_dir: str) -> dict:
    """전략검증 탭용 요약 페이로드. 데이터 파일이 없으면 ok=False로 응답."""
    path = os.path.join(base_dir, DATA_REL)
    if not os.path.exists(path):
        return {"ok": False, "error": f"데이터 파일 없음: {DATA_REL}"}
    rows = []
    with open(path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            out = dict(row)
            for k in _NUM_FIELDS:
                try:
                    out[k] = float(row[k])
                except (KeyError, TypeError, ValueError):
                    out[k] = None
            out["is_benchmark"] = row.get("code") == "069500"
            out["display_code"] = _display_code(row.get("code", ""))
            out["display_selection"] = _display_selection(row.get("selection", ""))
            rows.append(out)
    strategies = [r for r in rows if not r["is_benchmark"]]
    best = max(strategies, key=lambda r: (r["calmar"] is not None, r["calmar"])) if strategies else None
    return {
        "ok": True,
        "rows": rows,
        "best_code": best["code"] if best else None,
        "position": POSITION,
        "gates": GATES,
        "candidate": CANDIDATE,
        "principle": PRINCIPLE,
        "sample_curve": _load_sample_curve(base_dir),
        "source": DATA_REL,
    }


