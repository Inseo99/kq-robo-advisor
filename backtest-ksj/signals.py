"""신호 계산 (backtest-ksj): 모멘텀 선정 + 위험회피 게이트(t1/t2)."""

from __future__ import annotations

from dateutil.relativedelta import relativedelta
import numpy as np
import pandas as pd

import config as C


# ── 공통 헬퍼 ────────────────────────────────────────────────────────────
def asof_row(panel_ff: pd.DataFrame, date) -> pd.Series | None:
    """date 이하 최신 행(전종목) 반환. panel_ff는 ffill된 패널."""
    idx = panel_ff.index.asof(pd.Timestamp(date))
    if pd.isna(idx):
        return None
    return panel_ff.loc[idx]


def asof_val(series: pd.Series, date) -> float | None:
    idx = series.index.asof(pd.Timestamp(date))
    if pd.isna(idx):
        return None
    v = series.loc[idx]
    return float(v) if pd.notna(v) else None


def active_mask(adj_close_raw: pd.DataFrame, date, tol_days: int = 10) -> pd.Series:
    """date 직전 tol_days 이내 실거래(비결측)가 있는 종목 = 현재 상장/거래중."""
    d = pd.Timestamp(date)
    win = adj_close_raw.loc[d - pd.Timedelta(days=tol_days): d]
    if win.empty:
        return pd.Series(False, index=adj_close_raw.columns)
    return win.notna().any()


# ── 모멘텀 선정 ──────────────────────────────────────────────────────────
def momentum(close_ff: pd.DataFrame, eval_date) -> pd.Series:
    """12-1 모멘텀: P(t-1M)/P(t-12M) - 1."""
    d = pd.Timestamp(eval_date)
    p1 = asof_row(close_ff, d - relativedelta(months=C.MOM_SKIP_M))
    p12 = asof_row(close_ff, d - relativedelta(months=C.MOM_LOOKBACK_M))
    if p1 is None or p12 is None:
        return pd.Series(dtype=float)
    mom = p1 / p12 - 1.0
    mom = mom.replace([np.inf, -np.inf], np.nan).dropna()
    return mom


def select_top(close_ff, adj_close_raw, eval_date, pit_universe, n=None) -> list[str]:
    """PIT 유니버스 ∩ 현재거래중 종목에서 모멘텀 상위 n 코드."""
    n = n or C.TOP_N
    mom = momentum(close_ff, eval_date)
    if mom.empty:
        return []
    act = active_mask(adj_close_raw, eval_date)
    pit = set(pit_universe)
    cand = mom[[t for t in mom.index if t in pit and act.get(t, False)]]
    if cand.empty:
        return []
    return cand.nlargest(min(n, len(cand))).index.tolist()


# ── 위험회피 게이트 (t1 / t2) ────────────────────────────────────────────
def _resample(series: pd.Series, cadence: str) -> pd.Series:
    rule = "ME" if cadence == "M" else "W-FRI"
    return series.resample(rule).last().dropna()


def build_gate(cadence: str, variant: str, eval_dates: list,
               kodex_daily: pd.Series, vix: pd.Series | None,
               credit: pd.Series | None,
               sp500: pd.Series | None = None) -> tuple[dict, dict]:
    """eval_date -> 투자비중(1.0/0.0) 딕셔너리와 신호상세 딕셔너리 반환.

    variant: s1(항상 1.0) / s2(t1: 2개↑ 위험 -> 0.0) / s3(t2: MA 위 -> 1.0)

    t1 신호(s2): ① S&P500 < 9M MA  ② VIX > 18.6  ③ 미국 신용스프레드 z(5M) > 1.78
    → 3개 중 2개 이상 켜짐 → 주식비중 0%. (①은 S&P500, 없으면 KODEX200로 대체)
    """
    if variant == "s1":
        return {d: 1.0 for d in eval_dates}, {d: {} for d in eval_dates}

    k = _resample(kodex_daily, cadence)
    if cadence == "M":
        ma_t2 = k.rolling(C.T2_MA_M).mean()            # t2 10M (s3)
        trend_ma_win = C.T1_TREND_MA_M                 # t1① 9M
    else:
        ma_t2 = k.rolling(C.T2_MA_W).mean()            # 43주 (s3)
        trend_ma_win = C.T1_TREND_MA_W                 # 39주

    if variant == "s3":
        gate, detail = {}, {}
        for d in eval_dates:
            kc = asof_val(k, d)
            m = asof_val(ma_t2, d)
            invested = 1.0 if (kc is not None and m is not None and kc > m) else 0.0
            gate[d] = invested
            detail[d] = {"kodex": kc, "ma_t2": m, "invested": invested}
        return gate, detail

    # variant == 's2' : t1
    # ① 추세이탈: S&P500 종가 < 9M(39주) 이동평균. S&P500 미제공 시 KODEX200로 대체.
    trend_src = _resample(sp500, cadence) if sp500 is not None else k
    ma_trend = trend_src.rolling(trend_ma_win).mean()
    vix_r = _resample(vix, cadence) if vix is not None else None
    # ③ 신용 z: 일별 credit_spread에 ~5개월(105거래일) 롤링 z. 리밸런싱 주기와 무관하게 동일 기준.
    # (월간 리샘플 5개점은 z 상한≈1.79로 사실상 미발화하므로 일별 창으로 통일)
    cr_z = None
    if credit is not None:
        c = credit.sort_index()
        cr_z = (c - c.rolling(C.T1_CREDIT_Z_WIN_D).mean()) / c.rolling(C.T1_CREDIT_Z_WIN_D).std()

    gate, detail = {}, {}
    for d in eval_dates:
        pc = asof_val(trend_src, d)
        mt = asof_val(ma_trend, d)
        sig_trend = (pc is not None and mt is not None and pc < mt)          # ① S&P500<MA
        vv = asof_val(vix_r, d) if vix_r is not None else None
        sig_vix = (vv is not None and vv > C.T1_VIX_TH)                       # ② VIX>18.6
        zz = asof_val(cr_z, d) if cr_z is not None else None
        sig_credit = (zz is not None and zz > C.T1_CREDIT_Z_TH)              # ③ 신용 z>1.78
        n_on = int(sig_trend) + int(sig_vix) + int(sig_credit)
        risk_off = n_on >= C.T1_MIN_ON
        gate[d] = 0.0 if risk_off else 1.0
        detail[d] = {"trend": sig_trend, "vix": sig_vix, "credit": sig_credit,
                     "n_on": n_on, "risk_off": risk_off,
                     "sp500": pc, "sp500_ma": mt, "vix_val": vv, "credit_z": zz}
    return gate, detail
