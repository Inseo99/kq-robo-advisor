# -*- coding: utf-8 -*-
r"""N8 — 사후 확정 라벨 트랙 (ex-post confirmed track, NBER 기준순환일 방식의 2-트랙).

등록 (시도 1):
  실시간 트랙: data/regime/labels.csv (N5, 워크포워드·PiT) — 본 스크립트는 무접촉.
  확정 트랙:   위험 국면(스태그·디플레) 사이에 고립된 3개월 이하 골디락스 구간을
               사후적으로 양옆 위험 국면에 흡수. 수렴까지 반복 적용.
  규칙 근거:   시스템의 비대칭 원칙("위험 해제는 보수적으로")의 사후 확정판.
               스태그 섬(위험 진입 신호)과 리플레↔스태그 교대(완화 표현)는 불흡수.

  ** 용도 제한 (위반 시 미래 누수) **
  확정 라벨은 미래 3개월을 관찰해야 산출 가능한 비인과(non-causal) 시계열이다.
  회고 분석·논문 그림·국면 통계 서술에만 사용하고, 백테스트의 PiT 매칭·실시간 판정·
  분류기 학습에는 사용하지 않는다. 산출 파일명에 _expost_ 를 강제해 구분한다.

채점 (사전 고정):
  합격 = 앵커 6/6 유지 AND 창 일치율(10y vs 5y) 비열화(≥93.3%) AND 전환 감소
  기각 = 앵커 후퇴
"""
from __future__ import annotations

import os

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "data", "analysis_outputs")
RISK = {"스태그플레이션", "디플레이션"}


def _segments(s: pd.Series):
    segs, cur = [], None
    for t, v in s.items():
        if cur is None or cur[2] != v:
            if cur:
                segs.append(cur)
            cur = [t, t, v]
        else:
            cur[1] = t
    if cur:
        segs.append(cur)
    return segs


def confirm_labels(s: pd.Series) -> pd.Series:
    """위험 사이 ≤3개월 골디락스 섬 흡수 (수렴까지)."""
    s = s.copy()
    changed = True
    while changed:
        changed = False
        segs = _segments(s)
        for i in range(1, len(segs) - 1):
            (a1, b1, r1), (a2, b2, r2), (a3, b3, r3) = segs[i - 1], segs[i], segs[i + 1]
            blen = (b2.to_period("M") - a2.to_period("M")).n + 1
            if r2 == "골디락스" and blen <= 3 and r1 in RISK and r3 in RISK:
                s.loc[a2:b2] = r1
                changed = True
                break
    return s


def _metrics(s):
    segs = _segments(s)
    lens = [(b.to_period("M") - a.to_period("M")).n + 1 for a, b, _ in segs]
    return len(segs) - 1, sum(1 for l in lens if l <= 3), max(lens)


def main():
    results = {}
    for w in ["10y", "5y", "exp"]:
        src = pd.read_csv(os.path.join(OUT_DIR, f"labels_walkforward_{w}_n5.csv"),
                          index_col=0, parse_dates=True)["regime"].loc["2011-01-01":]
        conf = confirm_labels(src)
        conf.rename_axis("date").rename("regime").to_csv(
            os.path.join(OUT_DIR, f"labels_expost_confirmed_{w}.csv"),
            encoding="utf-8-sig", date_format="%Y-%m-%d")
        results[w] = conf
        if w == "10y":
            raw10, conf10 = src, conf

    tr0, s0, l0 = _metrics(raw10)
    tr1, s1, l1 = _metrics(conf10)
    print(f"[실시간(N5)]  전환 {tr0} · ≤3개월 {s0} · 최장 {l0}")
    print(f"[확정(N8)]    전환 {tr1} · ≤3개월 {s1} · 최장 {l1} · 변경 월 {(raw10 != conf10).sum()}")

    ok_map = {
        "2020-03": lambda x: x == "디플레이션",
        "2021-09": lambda x: x in ("리플레이션", "스태그플레이션"),
        "2021-10": lambda x: x in ("리플레이션", "스태그플레이션"),
        "2022-03": lambda x: x == "스태그플레이션",
        "2018-07": lambda x: x in ("스태그플레이션", "디플레이션"),
        "2020-12": lambda x: x in ("골디락스", "리플레이션"),
    }
    npass = 0
    for ym, fn in ok_map.items():
        v = conf10[conf10.index.strftime("%Y-%m") == ym].iloc[0]
        npass += fn(v)
    print(f"앵커: {npass}/6")

    idx = pd.date_range("2011-01-31", "2025-12-31", freq="ME")
    m = pd.DataFrame({k: v.reindex(idx) for k, v in results.items()}).dropna()
    ag = (m["10y"] == m["5y"]).mean()
    print(f"창 일치율(10y vs 5y): {ag:.1%} · 3창 전체 {(m.nunique(axis=1) == 1).mean():.1%}")
    print("\n[확정 트랙 구간 요약 (10y)]")
    for a, b, r in _segments(conf10):
        print(f"  {a:%Y-%m} ~ {b:%Y-%m}  {r}")
    verdict = "합격" if (npass == 6 and ag >= 0.933 and tr1 < tr0) else "불합격"
    print(f"\n[사전 고정 판정] 앵커 6/6 AND 일치율 ≥93.3% AND 전환 감소 => {verdict}")
    print("[용도 제한] 회고 분석·논문 그림 전용 — 백테스트 PiT 매칭·실시간 판정·분류기 학습 사용 금지")


if __name__ == "__main__":
    main()
