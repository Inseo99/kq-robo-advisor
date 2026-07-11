# -*- coding: utf-8 -*-
"""히트맵 '전략' 뷰 데이터 생성 — 6개 섹터·팩터 전략의 월별 수익률 + KODEX200.

옛 전략(quant/quant_s2/robo/ETF배분)을 대체해, 전략검증 탭의 6전략과 동일한
앱 근사 백테스트에서 월별 수익률 시계열을 뽑아 히트맵 CSV로 저장한다.

주의:
  - 반드시 로컬(실제 시장 데이터가 있는 환경)에서 실행한다. 세션/샘플 모드에서
    돌리면 무의미한 값이 나온다.
  - 이 데이터는 '앱 근사'이며, 정본 엔진(backtest-ksj, docx 표) 수치와는 다르다.
    전략검증 탭의 차트와 동일한 근사 기준이라 앱 내부 일관성은 유지된다.
  - 비용 기본값은 정본 방식(거래비용 25bps=편도 0.25%, 슬리피지 0)을 사용.

실행:
  python scripts/export_strategy_heatmap.py
  → tests/strategy_monthly_returns.csv 를 6전략+KODEX200 컬럼으로 덮어씀
  → 서버 재시작 + 브라우저 Ctrl+F5 후 히트맵 '전략' 탭 확인
"""
from __future__ import annotations

import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

OUT = os.path.join(ROOT, "tests", "strategy_monthly_returns.csv")
TOP_N = 5
REBALANCE = "M"
PERIOD = "12y"
COST_BPS = 25.0   # 정본: 편도 0.25%
SLIP_BPS = 0.0


def _equity_to_monthly_returns(dates, equity):
    s = pd.Series(equity, index=pd.to_datetime(dates)).astype(float)
    s = s.resample("ME").last().dropna()
    return s.pct_change().dropna() * 100.0


def main():
    import server  # 로컬 앱 컨텍스트 (실제 데이터 로드)

    result = server._run_strategy_backtest_all(TOP_N, REBALANCE, PERIOD, COST_BPS, SLIP_BPS)
    if result.get("error"):
        raise SystemExit(f"백테스트 실패: {result['error']}")

    # 표시명 매핑 (전략검증 탭과 동일)
    try:
        from kq_tool.backtest.sector_factor_summary import DISPLAY_CODES
    except Exception:
        DISPLAY_CODES = {}

    cols = {}
    for run in result.get("runs", []):
        if run.get("error"):
            continue
        code = run.get("strategy")
        label = DISPLAY_CODES.get(code, code)
        cols[label] = _equity_to_monthly_returns(run["dates"], run["equity"])

    # KODEX200 벤치마크
    if result.get("benchmark") and result.get("benchmark_dates"):
        cols["KODEX200 (벤치마크)"] = _equity_to_monthly_returns(
            result["benchmark_dates"], result["benchmark"])

    if not cols:
        raise SystemExit("생성된 전략 수익률이 없습니다.")

    df = pd.DataFrame(cols)
    df.index.name = "date"
    df = df.round(4)
    # 벤치마크를 맨 앞으로
    order = [c for c in df.columns if "벤치마크" in c] + [c for c in df.columns if "벤치마크" not in c]
    df = df[order]
    df.to_csv(OUT, encoding="utf-8-sig")
    print(f"저장 완료: {OUT}")
    print(f"기간: {df.index.min().date()} ~ {df.index.max().date()} · {len(df)}개월")
    print(f"전략 컬럼: {list(df.columns)}")


if __name__ == "__main__":
    main()
