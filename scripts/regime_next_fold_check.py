# -*- coding: utf-8 -*-
r"""regime-next N4 — 시기별 폴드 강건성 점검 (Purged K-Fold 하네스의 평가 모드).

용도: 후보 라벨의 개선이 특정 시기에 편중됐는지 검사. 5개 연속 폴드(각 36개월,
2011-01~2025-12)에서 창 일치율·전환 빈도·앵커를 폴드별로 비교한다.
(파라미터 선택이 없는 후보는 train/test 분리가 무의미하므로 선택 모드는 비활성 —
튜닝 파라미터가 있는 후보가 등장하면 embargo 포함 선택 모드를 이 파일에 추가한다.)
"""
from __future__ import annotations

import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "analysis_outputs")

CANDS = {
    "V2": ("labels_walkforward_10y_cpi.csv", "labels_walkforward_5y_cpi.csv"),
    "N3'": ("labels_walkforward_10y_n3.csv", "labels_walkforward_5y_n3.csv"),
    "N5": ("labels_walkforward_10y_n5.csv", "labels_walkforward_5y_n5.csv"),
}
ANCHORS = {
    "2020-03": lambda x: x == "디플레이션",
    "2021-09": lambda x: x in ("리플레이션", "스태그플레이션"),
    "2021-10": lambda x: x in ("리플레이션", "스태그플레이션"),
    "2022-03": lambda x: x == "스태그플레이션",
    "2018-07": lambda x: x in ("스태그플레이션", "디플레이션"),
    "2020-12": lambda x: x in ("골디락스", "리플레이션"),
}


def load(f):
    return pd.read_csv(os.path.join(OUT, f), index_col=0, parse_dates=True)["regime"]


def main():
    idx = pd.date_range("2011-01-31", "2025-12-31", freq="ME")
    folds = [idx[i * 36:(i + 1) * 36] for i in range(5)]
    print("폴드: " + " · ".join(f"F{k+1} {f[0]:%Y-%m}~{f[-1]:%Y-%m}" for k, f in enumerate(folds)))
    for name, (f10, f5) in CANDS.items():
        l10, l5 = load(f10).reindex(idx), load(f5).reindex(idx)
        rows = []
        for k, f in enumerate(folds):
            a = (l10[f] == l5[f]).mean() * 100
            tr = l10[f].dropna()
            n_tr = int((tr != tr.shift()).sum() - 1)
            anc = [(ym, fn(l10.get(pd.Timestamp(ym + "-01") + pd.offsets.MonthEnd(0))))
                   for ym, fn in ANCHORS.items()
                   if f[0] <= pd.Timestamp(ym + "-01") + pd.offsets.MonthEnd(0) <= f[-1]]
            anc_s = f"{sum(ok for _, ok in anc)}/{len(anc)}" if anc else "-"
            rows.append(f"F{k+1}: 일치 {a:.0f}% · 전환 {n_tr} · 앵커 {anc_s}")
        print(f"\n[{name}] " + " | ".join(rows))


if __name__ == "__main__":
    main()
