# -*- coding: utf-8 -*-
r"""J1 — 라벨 공백(홀드아웃) 구간의 잠정 판정(provisional judgment) 엔진.

문제 (사전 등록):
  화면의 현재 판정은 라벨 파일의 마지막 행을 90% 확률로 지속 표시한다(ui_payload).
  라벨이 끝난 뒤(현재 2026-01~)에는 신규 데이터가 판정에 반영될 수 없다 —
  2026 홀드아웃 대조와 확률 경화 분석이 동일하게 지적한 구조적 앵커링.

제안:
  라벨 종료 이후 월 t의 화면 판정을 "마지막 라벨 지속" 대신
  **잠정 판정 = 채택 라벨 규칙(N5)을 공표 지연 반영 데이터에 적용한 값**으로 교체.
  - 공표 지연은 docs/macro_publication_lags.md 의 적용 lag 준수 (cpi_yoy=1, gdp_qoq=2)
  - 라벨 파일에는 절대 쓰지 않음 (홀드아웃·확정 라벨 불가침) — 화면 표시 전용
  - 라벨이 존재하는 구간의 동작은 무변경

채점 (사전 고정):
  합격 = 2026-01~06 홀드아웃에서, 잠정 판정이 사후 워크포워드 기준 라벨과 일치하는
         월수가 현행 동작(마지막 라벨 '스태그' 지속 = 2/6)보다 많을 것.
  기록 = 공표 지연으로 인한 구조적 전환 인지 지연(개월)을 함께 보고.

시도: 1회 (J1). 산출: data/analysis_outputs/provisional_judgment_2026.csv
"""
from __future__ import annotations

import importlib.util
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "data", "analysis_outputs", "provisional_judgment_2026.csv")

LAGS = {"cpi": 1, "gdp": 2}   # docs/macro_publication_lags.md 적용 lag


def _load(modname, fname):
    spec = importlib.util.spec_from_file_location(modname, os.path.join(ROOT, "scripts", fname))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

wf = _load("wf", "walkforward_labels.py")
n5 = _load("n5mod", "regime_next_n5.py")


def lag_adjusted_slice(macro: dict, t: pd.Timestamp, window_m: int = 120) -> dict:
    """월말 t 시점에 실제로 공표돼 있는 데이터만 남긴 (t-창, t] 슬라이스."""
    t0 = t - pd.DateOffset(months=window_m)
    out = {}
    for k, v in macro.items():
        idx = pd.to_datetime(v.index)
        lag = LAGS.get(k, 0)
        cutoff = t - pd.DateOffset(months=lag)
        out[k] = v.loc[(idx > t0) & (idx <= cutoff)]
    return out


def main():
    macro = wf._load_macro_from_csv()
    labels = pd.read_csv(os.path.join(ROOT, "data", "regime", "labels.csv"),
                         encoding="utf-8-sig", parse_dates=["date"]).set_index("date")["regime"]
    label_end = labels.index.max()
    last_label = str(labels.iloc[-1])
    print(f"[현행 동작] 라벨 종료 {label_end:%Y-%m} 이후 화면 판정 = '{last_label}' 지속(90%)")

    # 사후 워크포워드 기준 (참조 — 기준월 표기 데이터, holdout_2026_comparison과 동일 방식)
    ref = {}
    ends = [pd.to_datetime(v.index).max() for v in macro.values()]
    months = pd.date_range(label_end + pd.offsets.MonthEnd(1), max(ends), freq="ME")
    rows = []
    for t in months:
        sliced_ref = wf._slice_macro(macro, t - pd.DateOffset(months=120), t)
        ref_lab = wf._extract_label(n5.make_labels_n5(sliced_ref), t)
        prov_lab = wf._extract_label(n5.make_labels_n5(lag_adjusted_slice(macro, t)), t)
        rows.append({
            "month": t.strftime("%Y-%m"),
            "현행(지속)": last_label,
            "잠정판정(공표지연 반영)": prov_lab,
            "사후 기준(참조)": ref_lab,
            "현행 일치": last_label == ref_lab,
            "잠정 일치": prov_lab == ref_lab,
        })
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(df.to_string(index=False))
    cur_hit = int(df["현행 일치"].sum())
    prov_hit = int(df["잠정 일치"].sum())
    n = len(df)
    print(f"\n[채점] 사후 기준 대비 일치: 현행 {cur_hit}/{n} vs 잠정 {prov_hit}/{n}")
    print(f"[판정] {'합격 — 잠정 판정이 현행보다 개선' if prov_hit > cur_hit else '불합격 — 개선 없음'}")
    print("[불변] 라벨 파일 무접촉 · 라벨 존재 구간 동작 무변경 · 공표 지연 lag 표 준수")


if __name__ == "__main__":
    main()
