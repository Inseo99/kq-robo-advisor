# -*- coding: utf-8 -*-
r"""국면별 자산군 수익률 차별성 검증 (기술 통계 - Discussion Q1 대응).

두 가지 매칭을 병행한다:
  동시점 매칭   - 국면의 경제적 특성 확인 (당월 라벨 × 당월 수익률).
                  주의: 당월 라벨은 당월 데이터를 반영하므로 거래 전략이 아니다.
  2개월 지연 매칭 - 실전 관측 가능 기준 (GDP 발표지연 2개월 반영, 모델 레이어의
                  label_delay_m=2와 동일). 라벨을 "알게 된 뒤"의 수익률.
출력: data/analysis_outputs/asset_returns_by_regime.csv
"""
import os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REG = ["골디락스", "리플레이션", "스태그플레이션", "디플레이션"]

ret = pd.read_csv(os.path.join(ROOT, "data/assets/monthly_returns.csv"), parse_dates=["date"]).set_index("date")
ret.index = ret.index.to_period("M").to_timestamp("M")
lab = pd.read_csv(os.path.join(ROOT, "data/regime/labels.csv"), encoding="utf-8-sig", parse_dates=["date"]).set_index("date")["regime"]
lab.index = lab.index.to_period("M").to_timestamp("M")

rows = []
for name, shift in [("동시점", 0), ("2개월지연", 2)]:
    df = ret.join(lab.shift(shift).rename("regime")).dropna()
    g = df.groupby("regime")
    mean = g.mean().reindex(REG) * 100
    sharpe = (g.mean() / g.std() * np.sqrt(12)).reindex(REG)
    n = g.size().reindex(REG)
    for r in REG:
        rows.append({"매칭": name, "국면": r, "n": int(n[r]),
                     **{f"{c}_평균%": round(mean.loc[r, c], 2) for c in ret.columns},
                     **{f"{c}_Sharpe": round(sharpe.loc[r, c], 2) for c in ret.columns}})
out = pd.DataFrame(rows)
out.to_csv(os.path.join(ROOT, "data/analysis_outputs/asset_returns_by_regime.csv"), index=False, encoding="utf-8-sig")
print(out.to_string(index=False))
print("\n[해석 지침] 동시점 표의 차별성 = 라벨의 경제적 타당성 / 지연 표와의 차이 = 관측 지연의 비용.")
print("[면책] 탐색적 기술 통계 - 유의성 주장 아님 (n<24 국면 주의).")
