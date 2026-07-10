# -*- coding: utf-8 -*-
r"""채택 라벨(N5) 월별 근거 테이블 — 각 국면 판정의 수치 근거를 투명하게 노출.

출력: data/analysis_outputs/label_evidence_table.csv
  판정월 · 분기 GDP QoQ · 창 중앙값(10y) · 성장축 · 분기 CPI YoY · 목표 T(t) · 물가축(완충 상태 포함) · 국면

무결성: 본 테이블의 국면 열이 채택 라벨(data/regime/labels.csv)과 180/180 일치해야
하며, 불일치 시 즉시 중단한다 (규칙 재구현의 드리프트 방지).
"""
from __future__ import annotations

import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "data", "analysis_outputs", "label_evidence_table.csv")
BUFFER = 0.5


def cpi_target(ts): return 2.0 if ts >= pd.Timestamp("2016-01-01") else 3.0


def main():
    gdp = pd.read_csv(os.path.join(ROOT, "data/macro/gdp_qoq.csv"), parse_dates=["date"]).set_index("date")["value"].resample("QE").last()
    cpi = pd.read_csv(os.path.join(ROOT, "data/macro/cpi_yoy.csv"), parse_dates=["date"]).set_index("date")["value"].resample("QE").last()
    adopted = pd.read_csv(os.path.join(ROOT, "data/regime/labels.csv"), encoding="utf-8-sig",
                          parse_dates=["date"]).set_index("date")["regime"]
    rows = []
    for t in pd.date_range("2011-01-31", "2025-12-31", freq="ME"):
        t0 = t - pd.DateOffset(months=120)
        g_win = gdp[(gdp.index > t0) & (gdp.index <= t)].dropna()
        c_win = cpi[(cpi.index > t0) & (cpi.index <= t)].dropna()
        med = g_win.median()
        # 물가축 상태 보유를 창 내에서 재현 (N5 규칙과 동일)
        curp = None
        for ts, v in c_win.items():
            T = cpi_target(ts)
            if curp is None:
                curp = bool(v > T)
            elif v > T:
                curp = True
            elif v < T - BUFFER:
                curp = False
        q_end = g_win.index[-1]
        g_val, c_val = g_win.iloc[-1], c_win.iloc[-1]
        hi_g = bool(g_val > med)
        T = cpi_target(c_win.index[-1])
        zone = "초과" if c_val > T else ("완충(유지)" if c_val >= T - BUFFER else "하회")
        regime = ("리플레이션" if hi_g and curp else "골디락스" if hi_g else
                  "스태그플레이션" if curp else "디플레이션")
        rows.append({
            "판정월": t.strftime("%Y-%m"), "분기": f"{q_end.year}Q{q_end.quarter}",
            "GDP_QoQ": round(float(g_val), 2), "창중앙값": round(float(med), 2),
            "성장축": "上" if hi_g else "下",
            "CPI_YoY": round(float(c_val), 2), "목표T": T, "목표대비": zone,
            "물가축": "上" if curp else "下", "국면": regime,
        })
    df = pd.DataFrame(rows)
    # 무결성 검증
    chk = adopted.reset_index()
    chk["판정월"] = chk["date"].dt.strftime("%Y-%m")
    merged = df.merge(chk[["판정월", "regime"]], on="판정월")
    mismatch = (merged["국면"] != merged["regime"]).sum()
    assert mismatch == 0, f"채택 라벨과 불일치 {mismatch}건 — 규칙 재구현 드리프트!"
    df.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"근거 테이블 {len(df)}행 저장 → {OUT}")
    print(f"[무결성] 채택 라벨과 180/180 일치 확인")
    for label, a, b in [("① 2011 스태그 15개월", "2011-03", "2012-05"),
                        ("② 2017 여름 스태그", "2017-05", "2017-09")]:
        print(f"\n== {label} ==")
        sub = df[(df["판정월"] >= a) & (df["판정월"] <= b)]
        print(sub.to_string(index=False))


if __name__ == "__main__":
    main()
