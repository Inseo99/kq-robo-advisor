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

DATA_REL = os.path.join("data", "strategy_validation", "sector_factor_backtest.csv")

POSITION = (
    "독립 백테스트 — 메인 추천 엔진과 분리된 전략검증 레이어이며 추천 엔진에는 "
    "반영되지 않습니다. 미국 섹터 모멘텀 상위 3섹터 → 한국 섹터 매핑 → 팩터 종목선정"
    "(7/5/3 동일가중), 월간 리밸런스, 거래비용 반영."
)
GATES = (
    "t1 — 미국 신호 3종 중 2개 이상 점등 시 전액 현금 "
    "(S&P500 < 9개월 이평 · VIX > 18.6 · 신용스프레드 z > 1.78) / "
    "t2 — KODEX200 종가 < 10개월 이평 시 전액 현금."
)
CANDIDATE = (
    "조건부 위험회피 결합은 기존 실험에서 방어 가능성을 보였으나, 현재 채택 라벨과 "
    "2개월 관측 지연 기준에서는 아직 재검증 전입니다. 따라서 성과 수치로 채택하지 "
    "않고, 후속 검증 후보로 분리합니다."
)
PRINCIPLE = (
    "성과가 좋아 보이는 전략이라도 동일한 검증 기준을 통과하기 전까지 "
    "추천 엔진에 반영하지 않습니다."
)

_NUM_FIELDS = ("cagr_pct", "mdd_pct", "sharpe", "calmar", "trades",
               "cost_total_pct", "cost_ann_pct")


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
        "source": DATA_REL,
    }
