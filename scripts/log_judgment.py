# -*- coding: utf-8 -*-
r"""운영 판정 월간 로거 — 홀드아웃 대조의 기록 공백 방지 (향후 연구 2번).

용도:
  실행 시점의 '화면 판정'을 월 1회 한 줄로 기록한다. 다음 홀드아웃 대조 때
  "실운영이 몇 월부터 무엇을 표시했는가"를 운영 로그로 확정하기 위함.
  2026 상반기 대조에서 기록 부재로 발표 시점 1개 관측만 가능했던 한계의 직접 처방.

동작:
  - 판정 산출은 앱과 동일 원리: 현재월이 라벨 범위 내면 확정 라벨,
    라벨 종료 이후면 잠정 판정(provisional_judgment, 공표 지연 반영) + '잠정' 표기
  - 월 1행 멱등: 같은 판정월 재실행 시 갱신하지 않음 (--force로 덮어쓰기)
  - 라벨 파일·채택 산출물 무접촉. 로그 파일에만 추가.

사용:
  python scripts\log_judgment.py            # 이번 달 기록 (이미 있으면 건너뜀)
  python scripts\log_judgment.py --force    # 이번 달 갱신
  (발표 당일 체크리스트(QA §8)에 월초 1회 실행을 추가하는 것을 권장)
"""
from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
LOG = os.path.join(ROOT, "data", "analysis_outputs", "operational_judgment_log.csv")


def _load(modname, fname):
    spec = importlib.util.spec_from_file_location(modname, os.path.join(ROOT, "scripts", fname))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="이번 달 행이 있어도 갱신")
    ap.add_argument("--asof", default=None, help="기준일 YYYY-MM-DD (기본: 오늘)")
    args = ap.parse_args()

    asof = pd.Timestamp(args.asof) if args.asof else pd.Timestamp(dt.date.today())
    month = asof.strftime("%Y-%m")

    labels = pd.read_csv(os.path.join(ROOT, "data", "regime", "labels.csv"),
                         encoding="utf-8-sig", parse_dates=["date"]).set_index("date")["regime"]
    label_end = labels.index.max()

    if asof <= label_end + pd.offsets.MonthEnd(0):
        # 라벨 범위 내 — 확정 라벨
        upto = labels[labels.index <= asof + pd.offsets.MonthEnd(0)]
        judgment, mode = str(upto.iloc[-1]), "확정 라벨"
        cpi_v = gdp_v = None
    else:
        # 라벨 공백 — 잠정 판정 (공표 지연 반영)
        wf = _load("wf", "walkforward_labels.py")
        n5 = _load("n5mod", "regime_next_n5.py")
        pj = _load("pj", "provisional_judgment.py")
        macro = wf._load_macro_from_csv()
        t = asof + pd.offsets.MonthEnd(0)
        sliced = pj.lag_adjusted_slice(macro, t)
        judgment, mode = wf._extract_label(n5.make_labels_n5(sliced), t), "잠정 (공표 지연 반영)"
        cpi_s = sliced["cpi"].iloc[:, 0].dropna()
        gdp_s = sliced["gdp"].iloc[:, 0].dropna()
        cpi_v = round(float(cpi_s.iloc[-1]), 2) if len(cpi_s) else None
        gdp_v = round(float(gdp_s.iloc[-1]), 2) if len(gdp_s) else None

    row = {
        "판정월": month,
        "기록일": asof.strftime("%Y-%m-%d"),
        "판정": judgment,
        "방식": mode,
        "라벨 종료월": label_end.strftime("%Y-%m"),
        "최신 공표 CPI": cpi_v,
        "최신 공표 GDP(QoQ)": gdp_v,
    }
    if os.path.exists(LOG):
        log = pd.read_csv(LOG, encoding="utf-8-sig", dtype={"판정월": str})
        if month in set(log["판정월"]) and not args.force:
            print(f"[skip] {month} 기록이 이미 있음 — 갱신하려면 --force")
            print(log.tail(3).to_string(index=False))
            return
        log = log[log["판정월"] != month]
        log = pd.concat([log, pd.DataFrame([row])], ignore_index=True)
    else:
        log = pd.DataFrame([row])
    log = log.sort_values("판정월")
    log.to_csv(LOG, index=False, encoding="utf-8-sig")
    print(f"[기록] {month}: {judgment} ({mode})")
    print(log.tail(3).to_string(index=False))


if __name__ == "__main__":
    main()
