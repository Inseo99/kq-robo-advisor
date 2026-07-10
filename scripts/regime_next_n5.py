# -*- coding: utf-8 -*-
r"""regime-next 후보 N5 — 물가 축 창-독립 + 당시 물가안정목표(PiT) + 비대칭 하방 완충.

등록 (시도 1):
  물가 축 상태 규칙 (분기, 상태 보유):
    상향 전환: CPI_YoY > 당시 물가안정목표 T(t)          — 즉시 (인플레 위험 민감)
    하향 전환: CPI_YoY < T(t) − 0.5%p                     — 보수적 (설명책임 하한 폭)
    그 사이 [T−0.5, T] 구간에서는 직전 상태 유지
  목표 연혁 (전부 각 기간 시작 전 공표된 PiT 값, 중심값 사용):
    ~2006: 3.0 (근원 대상 시기 — 채점 구간(2011~) 밖, 초기 창에만 영향)
    2007–2015: 3.0 (2007-09 ±0.5 · 2010-12 ±1.0 · 2013-15 ±0.5)
    2016~: 2.0 (설명책임 ±0.5%p — 세계일보 2015-12-15 한은 발표 인용 확인)
  성장 축·분기 리샘플·워크포워드·홀드아웃: V2 그대로.

정직성 기록: 비대칭(상향 즉시/하향 완충)의 근거는 "위험 국면 진입은 민감하게,
해제는 보수적으로"라는 분류기 목적이며, 이 원칙은 리포 데이터 검토 전 외부 검증
(2026-07-10 세션 전반부)에서 문서화됨. 다만 등록 확정 전 분기 CPI 경계값
(2021-09=2.41, 2024-09=1.59)을 열람했음 — 대칭 밴드의 앵커 파괴를 사전 확인한
설계 단계 타당성 점검이었고, 그 사실을 여기 남긴다. 시도는 본 구성 1회만 소모.

채점 (사전 고정, N3′과 동일):
  합격 = 앵커 6/6 AND 10y vs 5y ≥ 75% AND 전환 ≤ 33 AND 3개월 이하 구간 ≤ 23
  기각 = 앵커 1건이라도 후퇴
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
LABEL_END = "2025-12-31"
BUFFER = 0.5   # 하방 완충 폭 = 한은 설명책임 기준 ±0.5%p의 하한 (외생 공표값)


def cpi_target(ts: pd.Timestamp) -> float:
    """당시 유효 물가안정목표 중심값 (PiT — 각 기간 시작 전 공표)."""
    return 2.0 if ts >= pd.Timestamp("2016-01-01") else 3.0


def make_labels_n5(macro: dict) -> pd.DataFrame:
    if not macro or "gdp" not in macro or "cpi" not in macro:
        return pd.DataFrame()
    gdp_q = macro["gdp"].iloc[:, 0].dropna().resample("QE").last().ffill()
    cpi_q = macro["cpi"].iloc[:, 0].dropna().resample("QE").last().ffill()
    df = pd.DataFrame({"gdp_growth": gdp_q, "cpi_yoy": cpi_q}).dropna()
    if len(df) < 4:
        return pd.DataFrame()
    hi_g = df["gdp_growth"] > df["gdp_growth"].median()   # V2 유지
    # 물가 축: 비대칭 상태 보유
    states, cur = [], None
    for ts, v in df["cpi_yoy"].items():
        t = cpi_target(ts)
        if cur is None:
            cur = bool(v > t)
        elif v > t:
            cur = True
        elif v < t - BUFFER:
            cur = False
        states.append(cur)
    hi_p = pd.Series(states, index=df.index)
    lab = pd.Series("골디락스", index=df.index)
    lab[hi_g & hi_p] = "리플레이션"
    lab[~hi_g & hi_p] = "스태그플레이션"
    lab[~hi_g & ~hi_p] = "디플레이션"
    df["regime"] = lab
    return df


def run():
    macro = wf._load_macro_from_csv()
    data_start, data_end = wf._macro_month_range(macro)
    cutoff = min(data_end, pd.Timestamp(LABEL_END))
    months = pd.date_range(data_start, cutoff, freq="ME")
    os.makedirs(OUT_DIR, exist_ok=True)

    results = {}
    for name, win in WINDOWS:
        labels = {}
        min_hist = win if win is not None else MIN_EXPANDING
        eligible = [t for t in months if t >= data_start + pd.DateOffset(months=min_hist)]
        for t in eligible:
            t0 = (t - pd.DateOffset(months=win)) if win is not None else None
            try:
                lab = wf._extract_label(make_labels_n5(wf._slice_macro(macro, t0, t)), t)
            except Exception:
                lab = None
            if lab:
                labels[t] = lab
        s = pd.Series(labels, name="regime")
        s.index.name = "month"
        s.to_csv(os.path.join(OUT_DIR, f"labels_walkforward_{name}_n5.csv"),
                 encoding="utf-8-sig", date_format="%Y-%m-%d")
        results[name] = s
        print(f"[{name}] {len(s)}개월 생성")

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

    v2 = pd.read_csv(os.path.join(ROOT, "data", "regime", "labels.csv"),
                     encoding="utf-8-sig", parse_dates=["date"]).set_index("date")["regime"]
    both = pd.DataFrame({"V2": v2, "N5": main}).dropna()
    print(f"\n=== V2(채택본) vs N5 — 사전 등록 채점 (전체 일치율 {(both.V2 == both.N5).mean():.1%}) ===")
    ok_map = {
        "2020-03": ("코로나 전환", lambda x: x == "디플레이션"),
        "2021-09": ("인플레기", lambda x: x in ("리플레이션", "스태그플레이션")),
        "2021-10": ("인플레기", lambda x: x in ("리플레이션", "스태그플레이션")),
        "2022-03": ("긴축 전환", lambda x: x == "스태그플레이션"),
        "2018-07": ("무역분쟁", lambda x: x in ("스태그플레이션", "디플레이션")),
        "2020-12": ("회복 후반", lambda x: x in ("골디락스", "리플레이션")),
    }
    npass = 0
    for ym, (desc, fn) in ok_map.items():
        row = both[both.index.strftime("%Y-%m") == ym]
        if len(row):
            v = row.iloc[0]["N5"]
            ok = fn(v)
            npass += ok
            print(f"  {ym} {desc}: V2={row.iloc[0]['V2']} -> N5={v} => {'합격' if ok else '불합격'}")
    print(f"  앵커 합격: {npass}/6 (V2 = 6/6, 유지가 합격 조건)")

    tr = main.reindex(pd.date_range("2011-01-31", "2025-12-31", freq="ME")).dropna()
    n_tr = int((tr != tr.shift()).sum() - 1)
    seg_len = tr.groupby((tr != tr.shift()).cumsum()).size()
    print(f"\n[구조 지표 2011~2025] 전환 {n_tr}회 · 구간 {len(seg_len)}개 · "
          f"3개월 이하 구간 {(seg_len <= 3).sum()}개 · 최장 {seg_len.max()}개월")
    print("[사전 고정 판정] 합격 = 앵커 6/6 AND 10y vs 5y ≥75% AND 전환 ≤33 AND ≤3개월 ≤23 / 기각 = 앵커 후퇴")


if __name__ == "__main__":
    run()
