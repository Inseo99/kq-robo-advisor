"""데이터 접근 계층 (backtest-ksj).

원천:
  - data/cache/price_수정종가.parquet, price_수정시가.parquet  (수정주가, long)
  - data/cache/fin.parquet  (분기 재무 -> 시가총액 PIT)
  - data/cache/etf_assets_v2.parquet  (069500 = KODEX200, 2015-10~)
  - data/macro/kospi.csv  (2000~, 2015-10 이전 KODEX200 보강용)
  - backtest-ksj/data/vix.csv, credit_spread_us.csv  (사용자 제공, 없으면 None)

피벗 패널(date×ticker)은 backtest-ksj/data/에 parquet으로 캐시해 재실행을 가속한다.
"""

from __future__ import annotations

import os
import pandas as pd

import config as C


# ── 내부: long -> wide 피벗 + 캐시 ───────────────────────────────────────
def _pivot_cached(field: str) -> pd.DataFrame:
    """price_{field}.parquet(long)을 date×ticker 로 피벗하고 캐시."""
    cache_path = os.path.join(C.BT_DATA, f"panel_{field}.parquet")
    if os.path.exists(cache_path):
        return pd.read_parquet(cache_path)
    src = os.path.join(C.CACHE, f"price_{field}.parquet")
    df = pd.read_parquet(src)
    df["date"] = pd.to_datetime(df["date"])
    panel = df.pivot_table(index="date", columns="ticker", values="value", aggfunc="last")
    panel = panel.sort_index()
    panel.to_parquet(cache_path)
    return panel


def load_adj_close() -> pd.DataFrame:
    return _pivot_cached("수정종가")


def load_adj_open() -> pd.DataFrame:
    return _pivot_cached("수정시가")


# ── 시가총액 PIT ─────────────────────────────────────────────────────────
def load_mcap_history() -> pd.DataFrame:
    """분기 시가총액을 date×ticker 로 피벗 후 ffill (룩어헤드 방지용 PIT)."""
    cache_path = os.path.join(C.BT_DATA, "mcap_history.parquet")
    if os.path.exists(cache_path):
        return pd.read_parquet(cache_path)
    fin = pd.read_parquet(os.path.join(C.CACHE, "fin.parquet"))
    sub = fin[fin["아이템명"] == C.MCAP_KEY].copy()
    sub["date"] = pd.to_datetime(sub["date"])
    hist = sub.pivot_table(index="date", columns="ticker", values="value", aggfunc="last")
    hist = hist.sort_index().ffill()
    hist.to_parquet(cache_path)
    return hist


def top_mcap_at(mcap_hist: pd.DataFrame, date, n: int) -> list[str]:
    """date 시점(포함)까지 알려진 최신 시총 기준 상위 n 종목코드."""
    as_of = pd.Timestamp(date)
    avail = mcap_hist.index[mcap_hist.index <= as_of]
    if len(avail) == 0:
        return []
    row = pd.to_numeric(mcap_hist.loc[avail[-1]], errors="coerce").dropna()
    if row.empty:
        return []
    return row.nlargest(min(n, len(row))).index.tolist()


# ── KODEX200 (069500) 일별 시계열: 069500 + kospi.csv 보강 ────────────────
def load_kodex200_daily() -> pd.Series:
    """일별 KODEX200 종가. 2015-10 이후는 069500, 이전은 kospi.csv를 접합(스케일)."""
    etf = pd.read_parquet(os.path.join(C.CACHE, "etf_assets_v2.parquet"))
    k = etf["069500.KS"].copy()
    k.index = pd.to_datetime(k.index)
    k = k.sort_index().dropna()

    kospi = pd.read_csv(os.path.join(C.MACRO, "kospi.csv"))
    kospi["date"] = pd.to_datetime(kospi["date"])
    kospi = kospi.set_index("date")["value"].sort_index().dropna()

    splice_date = k.index.min()  # 069500 시작일
    # kospi를 splice_date에서 069500 값에 맞춰 스케일 후 이전 구간만 사용
    if splice_date in kospi.index:
        scale = k.loc[splice_date] / kospi.loc[splice_date]
    else:
        near = kospi.index[kospi.index <= splice_date]
        scale = k.iloc[0] / kospi.loc[near[-1]] if len(near) else 1.0
    kospi_scaled = kospi[kospi.index < splice_date] * scale
    combined = pd.concat([kospi_scaled, k]).sort_index()
    combined = combined[~combined.index.duplicated(keep="last")]
    combined.name = "kodex200"
    return combined


def load_kodex200_actual() -> pd.Series:
    """실제 069500 일별 종가 (벤치마크 성과용, 접합 없음)."""
    etf = pd.read_parquet(os.path.join(C.CACHE, "etf_assets_v2.parquet"))
    k = etf["069500.KS"].copy()
    k.index = pd.to_datetime(k.index)
    return k.sort_index().dropna()


# ── 외부 t1 데이터 (거시지표) ────────────────────────────────────────────
def _load_macro_ticker(filename: str, ticker: str) -> pd.Series | None:
    """backtest-ksj/data/의 long 포맷 거시 CSV(date,ticker,value)에서 단일 ticker 시계열 로드."""
    path = os.path.join(C.BT_DATA, filename)
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if "ticker" not in df.columns:
        return None
    df = df[df["ticker"] == ticker].copy()
    if df.empty:
        return None
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    s = (df.dropna(subset=["date", "value"])
           .set_index("date")["value"].sort_index())
    s = s[~s.index.duplicated(keep="last")]
    s.name = ticker
    return s


def _load_user_series(candidates: list[str], value_col_hint: str | None = None) -> pd.Series | None:
    """backtest-ksj/data/ 에서 후보 파일명 중 존재하는 CSV를 date-indexed Series로 로드."""
    for name in candidates:
        path = os.path.join(C.BT_DATA, name)
        if os.path.exists(path):
            df = pd.read_csv(path)
            cols = {c.lower(): c for c in df.columns}
            date_col = cols.get("date") or df.columns[0]
            val_col = None
            if value_col_hint and value_col_hint.lower() in cols:
                val_col = cols[value_col_hint.lower()]
            if val_col is None:
                val_col = cols.get("value") or (df.columns[1] if len(df.columns) > 1 else df.columns[-1])
            s = df[[date_col, val_col]].copy()
            s[date_col] = pd.to_datetime(s[date_col], errors="coerce")
            s = s.dropna(subset=[date_col])
            s[val_col] = pd.to_numeric(s[val_col], errors="coerce")
            s = s.dropna(subset=[val_col]).set_index(date_col)[val_col].sort_index()
            s.name = os.path.splitext(name)[0]
            return s
    return None


def load_sp500() -> pd.Series | None:
    """S&P500 지수 일별 종가 (market_2000_2026.csv, ticker=sp500). t1 신호① 추세이탈용."""
    return _load_macro_ticker("market_2000_2026.csv", "sp500")


def load_vix() -> pd.Series | None:
    """VIX 일별 (market_2000_2026.csv). 없으면 기존 단일-컬럼 CSV로 대체."""
    s = _load_macro_ticker("market_2000_2026.csv", "vix")
    if s is not None:
        return s
    return _load_user_series(["vix.csv", "VIX.csv", "vixcls.csv", "VIXCLS.csv"], "vix")


def load_credit_us() -> pd.Series | None:
    """미국 신용스프레드 일별 (liquidity_2000_2026.csv, ticker=credit_spread). 없으면 기존 CSV."""
    s = _load_macro_ticker("liquidity_2000_2026.csv", "credit_spread")
    if s is not None:
        return s
    return _load_user_series(
        ["credit_spread_us.csv", "us_credit_spread.csv", "credit_us.csv",
         "credit_spread.csv", "hy_oas.csv", "BAMLH0A0HYM2.csv"],
        "value",
    )


def t1_data_available() -> bool:
    return load_vix() is not None and load_credit_us() is not None
