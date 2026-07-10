# -*- coding: utf-8 -*-
r"""2026 홀드아웃 대조 (QA 시트 §9 로드맵 마지막 항목).

원칙:
  - data/regime/labels.csv 에는 어떤 2026 라벨도 쓰지 않는다 (홀드아웃 보존).
  - 산출은 data/analysis_outputs/holdout_2026_comparison.csv 뿐.
  - 각 월 t의 판정은 (t−창, t] 데이터만 사용 (워크포워드, 라벨 생성과 동일 규율).

비교 대상:
  V2  — 채택 이전 규칙 (성장·물가 모두 창 내 중앙값)
  N7  — 채택 규칙 (물가 창-독립 + 성장 강도 조건부 상향)
  실운영 판정 — 리포에 소급 기록이 없어 빈 칸으로 둠. 발표 이후 앱이 실제로 표시한
  판정(운영 로그·스크린샷)을 수기로 채워 대조하는 것이 §9의 원래 취지.
  (현 나우캐스트 산출물은 2025-12까지라 소급 대조 불가 — 재생성본은 '당시 판정'이 아님)
"""
from __future__ import annotations

import importlib.util
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "data", "analysis_outputs", "holdout_2026_comparison.csv")


def _load(modname, fname):
    spec = importlib.util.spec_from_file_location(modname, os.path.join(ROOT, "scripts", fname))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

wf = _load("wf", "walkforward_labels.py")
n7 = _load("n7mod", "regime_next_n7.py")
n5 = _load("n5mod", "regime_next_n5.py")


def main():
    macro = wf._load_macro_from_csv()
    # 계열별 꼬리 길이가 달라도 각 월 t는 그 시점까지 공표된 데이터만 쓰면 된다
    # (GDP 공표 지연 구간은 직전 분기 판정 유지 — 실운영과 동일 거동)
    ends = [pd.to_datetime(v.index).max() for v in macro.values()]
    data_end = max(ends)
    months = [t for t in pd.date_range("2026-01-31", data_end, freq="ME")]
    if not months:
        raise SystemExit("[stop] 2026년 데이터 없음")
    rows = []
    for t in months:
        t0 = t - pd.DateOffset(months=120)
        sliced = wf._slice_macro(macro, t0, t)
        v2 = wf._extract_label(wf._make_labels_v2(sliced), t)
        n5_10y = wf._extract_label(n5.make_labels_n5(sliced), t)
        n7_10y = wf._extract_label(n7.make_labels_n7(sliced), t)
        n7_5y = wf._extract_label(n7.make_labels_n7(wf._slice_macro(macro, t - pd.DateOffset(months=60), t)), t)
        n7_exp = wf._extract_label(n7.make_labels_n7(wf._slice_macro(macro, None, t)), t)
        cpi = macro["cpi"].iloc[:, 0].dropna()
        cpi_t = cpi[cpi.index <= t].iloc[-1] if len(cpi[cpi.index <= t]) else None
        gdp = macro["gdp"].iloc[:, 0].dropna()
        gdp_t = gdp[gdp.index <= t].iloc[-1] if len(gdp[gdp.index <= t]) else None
        rows.append({
            "month": t.strftime("%Y-%m"),
            "V2_10y": v2, "N5_10y(채택)": n5_10y, "N7_10y(참고)": n7_10y, "N7_5y": n7_5y, "N7_exp": n7_exp,
            "N7_3창일치": "예" if n7_10y == n7_5y == n7_exp else "아니오",
            "실운영판정(수기)": "",
            "CPI_YoY": round(float(cpi_t), 2) if cpi_t is not None else None,
            "GDP_QoQ(최근분기)": round(float(gdp_t), 2) if gdp_t is not None else None,
        })
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(df.to_string(index=False))
    print(f"\n저장: {OUT}")
    print("[확인] data/regime/labels.csv 는 건드리지 않음 — 홀드아웃 유지")
    ad = pd.read_csv(os.path.join(ROOT, "data", "regime", "labels.csv"), encoding="utf-8-sig")
    assert not ad["date"].astype(str).str.startswith("2026").any(), "홀드아웃 침범!"
    print("[검증] 채택 라벨에 2026 없음 ✓")


if __name__ == "__main__":
    main()
