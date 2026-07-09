"""scripts/fetch_etf_ohlc.py — 추천 유니버스 ETF 6종의 실제 시/고/저/종 수집
용도: reco_track의 오버나이트/장중 분해·고저 포락선 (실제 OHLC 전용 — 합성 금지)
소스: FinanceDataReader (fetch_prices.py와 동일 계열, KRX 수정가)
출력: data/prices/etf_ohlc.csv  (long: date,ticker,open,high,low,close)

사용: python scripts\fetch_etf_ohlc.py
이 파일이 없으면 reco_track의 장중 기능은 자동 비활성됩니다(안전한 기본값).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "prices" / "etf_ohlc.csv"
TICKERS = ["069500", "229200", "148070", "114260", "132030", "153130"]
START = "2014-01-01"


def main() -> None:
    import FinanceDataReader as fdr
    frames = []
    for t in TICKERS:
        df = fdr.DataReader(t, START)
        cols = {c.lower(): c for c in df.columns}
        need = ["open", "high", "low", "close"]
        if not all(k in cols for k in need):
            raise SystemExit(f"[stop] {t}: OHLC 컬럼 누락 — 실제 컬럼: {list(df.columns)}")
        out = df[[cols[k] for k in need]].copy()
        out.columns = need
        out.index.name = "date"
        out = out.reset_index()
        out.insert(1, "ticker", t)
        frames.append(out)
        print(f"[ok] {t}: {len(out)} rows ({out['date'].iloc[0].date()} ~ {out['date'].iloc[-1].date()})")
    panel = pd.concat(frames, ignore_index=True)
    # 무결성: low ≤ open,close ≤ high
    # 정제 1: 무거래일 시가 0 → 종가 대체 (2026-07 기준 0건, 미래 방어용)
    zero_open = panel["open"] <= 0
    panel.loc[zero_open, "open"] = panel.loc[zero_open, "close"]
    # 정제 2: 수정계수 불일치 미세 위반(229200 상장 초기 155행 실측) →
    #         고저를 시/종가가 포함되도록 클립. 포락선은 '한계' 정의라 보수적 확장.
    lo_fix = panel["low"] > panel[["open", "close"]].min(axis=1)
    hi_fix = panel["high"] < panel[["open", "close"]].max(axis=1)
    panel.loc[lo_fix, "low"] = panel.loc[lo_fix, ["low", "open", "close"]].min(axis=1)
    panel.loc[hi_fix, "high"] = panel.loc[hi_fix, ["high", "open", "close"]].max(axis=1)
    n_fixed = int(zero_open.sum() + (lo_fix | hi_fix).sum())
    if n_fixed / len(panel) > 0.02:
        raise SystemExit(f"[stop] 정제 비율 {n_fixed/len(panel):.1%} > 2% — 소스 자체 점검 필요")
    print(f"정제: 시가0 대체 {int(zero_open.sum())}행 · 고저 클립 {int((lo_fix | hi_fix).sum())}행 / 전체 {len(panel)}행")
    bad = panel[(panel["low"] > panel[["open", "close"]].min(axis=1)) |
                (panel["high"] < panel[["open", "close"]].max(axis=1))]
    if len(bad):
        raise SystemExit(f"[stop] 정제 후에도 위반 {len(bad)}행 잔존")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(OUT, index=False)
    print(f"saved: {OUT}  rows={len(panel)}  (low≤open,close≤high 정합 통과)")
    print("next: 서버 재시작 → 히트맵 하단 모의 트랙에 장중 분해·포락선 활성 확인")


if __name__ == "__main__":
    main()
