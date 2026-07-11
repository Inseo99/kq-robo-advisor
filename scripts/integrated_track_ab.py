# -*- coding: utf-8 -*-
r"""통합 성과 트랙 A/B — 국면 배분 × 스크리너 코어-위성 (사전등록 실행기).

세 조각을 하나로 합친 최종 결과물:
  ① 국면 → ERC v1 자산배분 (추천 규칙 모의 트랙)
  ② 스크리너 최고 전략 (전략검증)
  ③ 코어-위성 결합 (①의 주식 버킷을 ②로 일부 대체)

트랙 정의 (동일 국면 배분·동일 기간·동일 비용):
  A (기준): 주식 버킷 = 기존 주식 ETF(KODEX200+코스닥150) 100%
  B (처치): 주식 버킷 = 코어 70%(기존 주식 ETF) + 위성 30%(스크리너 최고 전략)
  → B−A = 매월 (주식비중 × 0.30 × (위성수익률 − 코어수익률))

입력:
  data/regime/labels.csv                         국면 라벨(월)
  data/analysis_outputs/regime_erc_v1_weights.csv 국면별 ERC 자산군 비중
  tests/asset_monthly_returns.csv                ETF 월수익률 (저장소 포함 → A 계산 가능)
  [선택] --satellite <csv>  위성(스크리너 최고 전략) 월수익률 1열
         (없으면 Track A만 계산. 로컬에서 scripts/export_strategy_heatmap.py로 생성 가능)

사전등록 규율(docs/통합OOS검증_사전등록.md):
  - 위성 전략은 학습 구간에서 사전 선택(전체기간 최고 사용 금지 = 룩어헤드).
  - 위성은 게이트 없는 상시투자 버전. 방어는 국면 배분이 담당.
  - 채점: OOS Calmar. B ≥ A×1.05 AND MDD 비악화면 위성 승격.
  - 관측 지연: --lag (기본 1개월; 엄격 기준 2).

주의: tests/asset_monthly_returns.csv는 앱 근사 데이터. 정본 결론은 시총300·PIT 재무
      백테스터에서 재실행할 것. 본 스크립트는 통합 로직의 재현·구조 검증용.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LABELS = os.path.join(ROOT, "data", "regime", "labels.csv")
ERC = os.path.join(ROOT, "data", "analysis_outputs", "regime_erc_v1_weights.csv")
ASSETS = os.path.join(ROOT, "tests", "asset_monthly_returns.csv")

STOCK_ETFS = ["069500.KS", "229200.KS"]
CLASS_ETFS = {
    "stocks": ["069500.KS", "229200.KS"],
    "bonds": ["148070.KS", "114260.KS"],
    "gold": ["132030.KS"],
    "cash": ["153130.KS"],
}
CORE_RATIO, SAT_RATIO = 0.70, 0.30
RF = 0.035


def _metrics(monthly_ret: pd.Series, freq: int = 12) -> dict:
    r = monthly_ret.dropna()
    if len(r) < 3:
        return {}
    nav = (1 + r).cumprod()
    years = len(r) / freq
    total = float(nav.iloc[-1])
    cagr = total ** (1 / years) - 1
    vol = float(r.std() * np.sqrt(freq))
    sharpe = (cagr - RF) / vol if vol > 0 else 0.0
    mdd = float(((nav - nav.cummax()) / nav.cummax()).min())
    calmar = cagr / abs(mdd) if abs(mdd) > 1e-9 else 0.0
    return {"CAGR%": round(cagr * 100, 2), "MDD%": round(mdd * 100, 2),
            "Sharpe": round(sharpe, 3), "Calmar": round(calmar, 3),
            "누적%": round((total - 1) * 100, 2), "개월": len(r)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--satellite", help="위성 월수익률 CSV (export_strategy_heatmap.py 산출물 그대로 가능)")
    ap.add_argument("--sat-col", help="위성으로 쓸 컬럼명 (다중 컬럼 CSV 필수 — 예: 'M1 중형성장주'). "
                                      "미지정 시 컬럼 목록을 출력하고 중단")
    ap.add_argument("--lag", type=int, default=1, help="국면 관측 지연(개월). 기본 1, 엄격 2")
    ap.add_argument("--oos-start", default="2020-01", help="OOS 구간 시작(YYYY-MM)")
    args = ap.parse_args()

    labels = pd.read_csv(LABELS, encoding="utf-8-sig", parse_dates=["date"]).set_index("date")["regime"]
    labels.index = labels.index.to_period("M")
    erc = pd.read_csv(ERC, index_col=0, encoding="utf-8-sig")
    ret = pd.read_csv(ASSETS, parse_dates=["Date"]).set_index("Date")
    ret.index = ret.index.to_period("M")

    lagged_regime = labels.shift(args.lag)  # t월 배분에 t-lag월 확정 국면 사용
    rows_a, rows_b = [], []
    idx = []

    sat = None
    if args.satellite and os.path.exists(args.satellite):
        sdf = pd.read_csv(args.satellite, encoding="utf-8-sig")
        dcol = next((c for c in sdf.columns if "date" in c.lower() or "월" in c), sdf.columns[0])
        value_cols = [c for c in sdf.columns if c != dcol]
        if len(value_cols) == 1:
            vcol = value_cols[0]
        elif args.sat_col and args.sat_col in sdf.columns:
            vcol = args.sat_col
        else:
            # 다중 컬럼인데 미지정 → 조용한 오류 방지, 컬럼 목록 출력 후 중단
            raise SystemExit(
                "위성 CSV에 컬럼이 여러 개입니다. --sat-col 로 명시하세요.\n"
                f"  사용 가능 컬럼: {value_cols}\n"
                "  예: --sat-col 'M1 중형성장주'  (검증탭 Calmar 최우수 전략)")
        sat = pd.Series(pd.to_numeric(sdf[vcol], errors="coerce").values,
                        index=pd.to_datetime(sdf[dcol]).dt.to_period("M"))
        print(f"[위성] '{vcol}' 컬럼 사용")

    for m in ret.index:
        reg = lagged_regime.get(m)
        if reg is None or reg not in erc.index:
            continue
        cw = erc.loc[reg]  # stocks/bonds/gold/cash
        # 비주식 클래스 수익 (A·B 동일)
        nonstock = 0.0
        for cls in ("bonds", "gold", "cash"):
            etfs = CLASS_ETFS[cls]
            cls_ret = np.mean([ret.loc[m, e] for e in etfs if e in ret.columns])
            nonstock += float(cw[cls]) * cls_ret
        # 주식 클래스
        Ws = float(cw["stocks"])
        core_ret = np.mean([ret.loc[m, e] for e in STOCK_ETFS if e in ret.columns])
        rA = nonstock + Ws * core_ret
        if sat is not None and m in sat.index and not np.isnan(sat.get(m, np.nan)):
            sat_ret = float(sat.get(m))
            stock_b = CORE_RATIO * core_ret + SAT_RATIO * sat_ret
            rB = nonstock + Ws * stock_b
        else:
            rB = np.nan
        idx.append(m); rows_a.append(rA); rows_b.append(rB)

    A = pd.Series(rows_a, index=pd.PeriodIndex(idx, freq="M"))
    B = pd.Series(rows_b, index=pd.PeriodIndex(idx, freq="M"))
    oos = pd.Period(args.oos_start, freq="M")

    print(f"통합 트랙 (lag={args.lag}개월 · 기간 {A.index[0]}~{A.index[-1]})")
    print(f"\n[Track A · 국면→ETF 배분, 주식=KODEX200 중심 패시브]")
    print("  전체:", _metrics(A))
    print("  OOS :", _metrics(A[A.index >= oos]))
    if B.notna().sum() >= 3:
        print(f"\n[Track B · 코어-위성 70:30]")
        print("  전체:", _metrics(B))
        print("  OOS :", _metrics(B[B.index >= oos]))
        mA, mB = _metrics(A[A.index >= oos]), _metrics(B[B.index >= oos])
        promote = (mB.get("Calmar", 0) >= mA.get("Calmar", 0) * 1.05
                   and mB.get("MDD%", -99) >= mA.get("MDD%", -99) - 2.0)
        print(f"\n[사전등록 판정] OOS Calmar A {mA.get('Calmar')} vs B {mB.get('Calmar')} · "
              f"{'위성 승격 후보' if promote else '미승격 — 스크리너 후보 유지'}")
    else:
        print("\n[Track B] 위성 월수익률 미입력 — --satellite <csv> 로 스크리너 최고 전략 "
              "월수익률을 넣으면 코어-위성 통합 트랙이 계산됩니다.")
        print("  (로컬: python scripts/export_strategy_heatmap.py 로 6전략 월수익률 생성 후,")
        print("   그중 최고 전략 열을 뽑아 date,ret 형식으로 전달)")
    print("\n[주의] 앱 근사 데이터 기준. 정본 결론은 시총300·PIT 재무 백테스터에서 재실행.")


if __name__ == "__main__":
    main()
