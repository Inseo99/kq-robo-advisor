# -*- coding: utf-8 -*-
r"""regime-next 후보 N1 — 성장 축 월별 지표 병용 (GDP + 선행지수 z-블렌드).

사전 등록 (regime_next_사전등록_초안.md, 시도 1):
  성장 축: z_gdp + z_lead > 0  (z = (x − 창 내 중앙값) / 창 내 표준편차, 동일 가중 — 튜닝 없음)
           GDP(분기, 전기비)는 월간 ffill, 선행지수순환변동치는 월간 원계열
  물가 축: CPI YoY > 창 내 중앙값 (V2 규칙, 해상도만 월간)
  해상도: 월간 (분기 각짐 = 채터링 근본 원인 → 해소 목표)
  워크포워드: 매월 t, (t−창, t] 절단 데이터만 사용. 홀드아웃(2026) 불가침.

채점 (사전 고정):
  합격 = 사전 등록 앵커 6종 비열화 + 전환 20±8 + 3개월 이하 구간 < 23(V2)
         + 창 민감도(10y vs 5y) ≥ 68.3%(V2)
  기각 = 앵커 1건이라도 후퇴, 또는 전환 ≥ 30회

산출물: data/analysis_outputs/labels_walkforward_{10y,5y,exp}_n1.csv (채택 라벨 무수정)
"""
from __future__ import annotations

import importlib.util
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
OUT_DIR = os.path.join(ROOT, "data", "analysis_outputs")

_spec = importlib.util.spec_from_file_location(
    "wf", os.path.join(ROOT, "scripts", "walkforward_labels.py"))
wf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(wf)

WINDOWS = [("10y", 120), ("5y", 60), ("exp", None)]
MIN_EXPANDING = 60
LABEL_END = "2025-12-31"   # 홀드아웃 유지


def _z(s: pd.Series) -> pd.Series:
    sd = s.std(ddof=1)
    if not sd or pd.isna(sd):
        return s * 0.0
    return (s - s.median()) / sd


def make_labels_n1(macro: dict) -> pd.DataFrame:
    """성장 = z(GDP)+z(선행) > 0 (월간), 물가 = CPI > 중앙값 (월간). 창 내 통계만 사용."""
    if not macro or "gdp" not in macro or "cpi" not in macro or "growth_m" not in macro:
        return pd.DataFrame()
    gm = macro["growth_m"]
    if "leading_index_cycle" not in gm.columns:
        return pd.DataFrame()
    gdp = macro["gdp"].iloc[:, 0].dropna().resample("ME").last().ffill()
    lead = gm["leading_index_cycle"].dropna().resample("ME").last().ffill()
    cpi = macro["cpi"].iloc[:, 0].dropna().resample("ME").last().ffill()
    df = pd.DataFrame({"gdp": gdp, "lead": lead, "cpi": cpi}).dropna()
    if len(df) < 12:
        return pd.DataFrame()
    hi_g = (_z(df["gdp"]) + _z(df["lead"])) > 0
    hi_p = df["cpi"] > df["cpi"].median()
    lab = pd.Series("골디락스", index=df.index)
    lab[hi_g & hi_p] = "리플레이션"
    lab[~hi_g & hi_p] = "스태그플레이션"
    lab[~hi_g & ~hi_p] = "디플레이션"
    df["regime"] = lab
    return df


def run():
    macro = wf._load_macro_from_csv()
    if not macro:
        raise SystemExit("[stop] 매크로 데이터 없음")
    data_start, data_end = wf._macro_month_range(macro)
    cutoff = min(data_end, pd.Timestamp(LABEL_END))
    months = pd.date_range(data_start, cutoff, freq="ME")
    if cutoff < data_end:
        print(f"[홀드아웃] 라벨 생성은 {cutoff:%Y-%m}까지")
    os.makedirs(OUT_DIR, exist_ok=True)

    results = {}
    for name, win in WINDOWS:
        labels = {}
        min_hist = win if win is not None else MIN_EXPANDING
        eligible = [t for t in months if t >= data_start + pd.DateOffset(months=min_hist)]
        print(f"[{name}] {len(eligible)}개월 생성 중 (시작 {eligible[0]:%Y-%m})...")
        for t in eligible:
            t0 = (t - pd.DateOffset(months=win)) if win is not None else None
            try:
                lab = wf._extract_label(make_labels_n1(wf._slice_macro(macro, t0, t)), t)
            except Exception:
                lab = None
            if lab:
                labels[t] = lab
        s = pd.Series(labels, name="regime")
        s.index.name = "month"
        path = os.path.join(OUT_DIR, f"labels_walkforward_{name}_n1.csv")
        s.to_csv(path, encoding="utf-8-sig", date_format="%Y-%m-%d")
        results[name] = s
        print(f"[{name}] {len(s)}개월 → {path}")

    names = [n for n, _ in WINDOWS]
    merged = pd.DataFrame(results).dropna()
    print("\n=== 창 선택 민감도 (공통 구간 {}~{}) ===".format(
        merged.index[0].strftime("%Y-%m"), merged.index[-1].strftime("%Y-%m")))
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            print(f"  {names[i]} vs {names[j]}: 일치율 {(merged[names[i]] == merged[names[j]]).mean():.1%}")
    print(f"  3창 전체 일치: {(merged.nunique(axis=1) == 1).mean():.1%}")

    print("\n=== 위기 구간 채점 (기대: 디플레/스태그 계열) ===")
    for period, desc in [("2008-09", "금융위기"), ("2020-0", "코로나"), ("2022", "긴축 하락장")]:
        for name in names:
            s = results[name]
            hit = s[s.index.strftime("%Y-%m").str.startswith(period)]
            if len(hit):
                print(f"  {desc} [{name}]: " + ", ".join(f"{d:%Y-%m}={v}" for d, v in list(hit.items())[:6]))

    main = results["10y"]
    print("\n=== 주 라벨(10y) 구간 요약 ===")
    cur, start, prev = None, None, None
    for dt, lab in main.items():
        if lab != cur:
            if cur is not None:
                print(f"  {start:%Y-%m} ~ {prev:%Y-%m}  {cur}")
            cur, start = lab, dt
        prev = dt
    print(f"  {start:%Y-%m} ~ {prev:%Y-%m}  {cur}")

    # V2(채택본) 대비 사전 등록 채점
    v2 = pd.read_csv(os.path.join(ROOT, "data", "regime", "labels.csv"),
                     encoding="utf-8-sig", parse_dates=["date"]).set_index("date")["regime"]
    both = pd.DataFrame({"기준(V2 채택본)": v2, "N1": main}).dropna()
    print(f"\n=== V2(채택본) vs N1 — 사전 등록 채점 (전체 일치율 {(both.iloc[:, 0] == both.iloc[:, 1]).mean():.1%}) ===")
    checks = [("2020-03", "코로나 전환", "디플레이션 유지가 합격"),
              ("2021-09", "인플레기", "리플레/스태그 유지가 합격"),
              ("2021-10", "인플레기", "동일"),
              ("2022-03", "긴축 전환", "스태그 유지가 합격"),
              ("2018-07", "무역분쟁", "위험 국면(스태그/디플레) 진입이면 개선"),
              ("2020-12", "회복 후반", "골디락스/리플레면 합격")]
    ok_map = {
        "2020-03": lambda x: x == "디플레이션",
        "2021-09": lambda x: x in ("리플레이션", "스태그플레이션"),
        "2021-10": lambda x: x in ("리플레이션", "스태그플레이션"),
        "2022-03": lambda x: x == "스태그플레이션",
        "2018-07": lambda x: x in ("스태그플레이션", "디플레이션"),
        "2020-12": lambda x: x in ("골디락스", "리플레이션"),
    }
    npass = 0
    for ym, desc, crit in checks:
        row = both[both.index.strftime("%Y-%m") == ym]
        if len(row):
            v = row.iloc[0, 1]
            ok = ok_map[ym](v)
            npass += ok
            print(f"  {ym} {desc}: 기준={row.iloc[0, 0]} -> N1={v}  [{crit}] => {'합격' if ok else '불합격'}")
    print(f"  앵커 합격: {npass}/6 (동일 척도에서 V2 채택본은 6/6 — 비열화가 합격 조건)")

    tr = main
    n_tr = int((tr != tr.shift()).sum() - 1)
    seg_len = tr.groupby((tr != tr.shift()).cumsum()).size()
    print(f"\n[구조 지표] 전환 {n_tr}회 · 구간 {len(seg_len)}개 · 3개월 이하 구간 {(seg_len <= 3).sum()}개 · 최장 {seg_len.max()}개월")
    print("[사전 고정 판정 기준] 앵커 6종 비열화 · 전환 20±8 · ≤3개월 구간 <23 · 10y vs 5y ≥68.3% / 기각: 앵커 후퇴 또는 전환 ≥30")


if __name__ == "__main__":
    run()
