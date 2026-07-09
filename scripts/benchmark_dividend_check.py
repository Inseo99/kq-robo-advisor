"""scripts/benchmark_dividend_check.py — KODEX200 분배금 반영 여부 판별 + TR 근사 생성
(로컬 실행, pykrx 필요)

판별: KODEX200 가격 CAGR vs KOSPI200 지수 CAGR 비교
  차이 ≈ 0      → 양쪽 다 가격 기준 (둘 다 배당 미반영: 벤치마크가 연 ~1.5-2%p 약함)
  ETF가 +1.5%p↑ → ETF 수정주가에 분배금 재투자 반영됨 (그대로 사용 가능)

옵션: --make-tr  가격 기준일 경우, KOSPI200 배당수익률(연 시계열)을 일할 가산한
  TR 근사 벤치마크 CSV 생성 → backtest-ksj 벤치마크로 교체 사용

사용: python scripts/benchmark_dividend_check.py --start 2014-01-01 --end 2026-05-31
"""
from __future__ import annotations
import argparse
import pandas as pd

# KOSPI200 연평균 배당수익률 근사 (KRX 공시 기반 대략치, TR 근사용)
DIV_YIELD_BY_YEAR = {
    2014: 0.011, 2015: 0.013, 2016: 0.015, 2017: 0.015, 2018: 0.019,
    2019: 0.020, 2020: 0.018, 2021: 0.017, 2022: 0.022, 2023: 0.020,
    2024: 0.021, 2025: 0.019, 2026: 0.019,
}


def cagr(series: pd.Series) -> float:
    yrs = (series.index[-1] - series.index[0]).days / 365.25
    return float((series.iloc[-1] / series.iloc[0]) ** (1 / yrs) - 1)


def main(start: str, end: str, make_tr: bool):
    from pykrx import stock
    s, e = start.replace("-", ""), end.replace("-", "")
    etf = stock.get_etf_ohlcv_by_date(s, e, "069500")["종가"]
    etf.index = pd.to_datetime(etf.index)
    k200 = stock.get_index_ohlcv_by_date(s, e, "1028")["종가"]  # KOSPI200
    k200.index = pd.to_datetime(k200.index)

    c_etf, c_idx = cagr(etf), cagr(k200)
    gap = c_etf - c_idx
    print(f"KODEX200 가격 CAGR : {c_etf:.2%}")
    print(f"KOSPI200 지수 CAGR : {c_idx:.2%}")
    print(f"차이               : {gap:+.2%}p/년")
    if gap > 0.012:
        print("\n판정: ETF 수정주가에 분배금 재투자가 반영된 것으로 보임")
        print("→ 현행 벤치마크 유지. README에 '분배금 반영 수정주가 기준' 1줄 명시.")
    elif abs(gap) <= 0.012:
        print("\n판정: 가격 기준(분배금 미반영) 가능성 높음")
        print("→ 벤치마크가 연 ~1.5-2%p 약함 = 전략 초과수익이 그만큼 과대.")
        print("→ --make-tr 로 TR 근사 벤치마크 생성 후 교체 권장,")
        print("  또는 최소한 발표 각주: '벤치마크는 가격수익률 기준(배당 제외),"
              " 전략 초과수익은 배당수익률(연 ~1.5-2%p)만큼 보수적으로 해석 필요'")
    else:
        print("\n판정: ETF가 지수 대비 크게 낮음 — 데이터 이상(수정주가 미적용 등) 점검 필요")

    if make_tr:
        daily = etf.pct_change().fillna(0.0)
        add = daily.index.year.map(lambda y: DIV_YIELD_BY_YEAR.get(y, 0.019)) / 252
        tr = (1 + daily + pd.Series(add, index=daily.index)).cumprod() * etf.iloc[0]
        out = "backtest-ksj/data/kodex200_tr_approx.csv"
        tr.rename("kodex200_tr").to_csv(out, encoding="utf-8-sig")
        print(f"\nTR 근사 벤치마크 저장: {out}")
        print(f"TR 근사 CAGR: {cagr(tr):.2%} (가격 대비 +{cagr(tr)-c_etf:.2%}p)")
        print("주의: 연평균 배당수익률의 일할 가산 근사임 — 리포트에 근사 방식 명시")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2014-01-01")
    ap.add_argument("--end", default="2026-05-31")
    ap.add_argument("--make-tr", action="store_true")
    a = ap.parse_args()
    main(a.start, a.end, a.make_tr)
