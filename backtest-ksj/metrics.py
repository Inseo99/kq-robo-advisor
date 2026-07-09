"""성과지표 (backtest-ksj): CAGR / MDD / Sharpe / Calmar + 매매·비용 집계."""

from __future__ import annotations

import numpy as np
import pandas as pd

import config as C


def _window_slice(df: pd.DataFrame, start, end, date_col="exit") -> pd.DataFrame:
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    return df[(df[date_col] >= s) & (df[date_col] <= e)].copy()


def mdd_from_returns(returns: pd.Series) -> float:
    if len(returns) == 0:
        return 0.0
    eq = (1.0 + returns).cumprod()
    dd = eq / eq.cummax() - 1.0
    return float(dd.min())


def strategy_window_metrics(df: pd.DataFrame, cadence: str, start, end) -> dict:
    """전략 기간 레코드 df에서 [start,end] 창(수익실현일 기준) 지표."""
    sub = _window_slice(df, start, end, "exit")
    ppy = C.PERIODS_PER_YEAR[cadence]
    cash_period = C.CASH_MONTHLY if cadence == "M" else C.CASH_WEEKLY
    n = len(sub)
    if n == 0:
        return {"n_periods": 0, "cagr": np.nan, "mdd": np.nan, "sharpe": np.nan,
                "calmar": np.nan, "trades": 0, "cost_total_pct": 0.0,
                "cost_ann_pct": 0.0, "total_return": np.nan, "years": 0.0,
                "n_rebal_invested": 0}
    ret = sub["net"].reset_index(drop=True)
    total = float((1.0 + ret).prod() - 1.0)
    years = (pd.Timestamp(sub["exit"].iloc[-1]) - pd.Timestamp(sub["entry"].iloc[0])).days / 365.25
    years = max(years, 1e-9)
    cagr = (1.0 + total) ** (1.0 / years) - 1.0
    mdd = mdd_from_returns(ret)
    excess = ret - cash_period
    sd = float(ret.std(ddof=1)) if n > 1 else 0.0
    sharpe = float(excess.mean() / sd * np.sqrt(ppy)) if sd > 0 else np.nan
    calmar = float(cagr / abs(mdd)) if mdd < 0 else np.nan
    trades = int(sub["buys"].sum() + sub["sells"].sum())
    cost_total = float(sub["cost"].sum())            # 누적 비용(수익률 차감분 합, 근사)
    cost_ann = cost_total / years
    return {
        "n_periods": n, "cagr": cagr, "mdd": mdd, "sharpe": sharpe, "calmar": calmar,
        "trades": trades, "cost_total_pct": cost_total * 100, "cost_ann_pct": cost_ann * 100,
        "total_return": total, "years": years,
        "n_rebal_invested": int((sub["n_hold"] > 0).sum()),
    }


def benchmark_window_metrics(bench_daily: pd.Series, start, end) -> dict:
    """일별 벤치마크(069500) 종가에서 [start,end] 지표."""
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    px = bench_daily[(bench_daily.index >= s) & (bench_daily.index <= e)].dropna()
    if len(px) < 2:
        return {"n_periods": len(px), "cagr": np.nan, "mdd": np.nan, "sharpe": np.nan,
                "calmar": np.nan, "trades": 0, "cost_total_pct": 0.0, "cost_ann_pct": 0.0,
                "total_return": np.nan, "years": 0.0, "n_rebal_invested": 0}
    ret = px.pct_change().dropna()
    total = float(px.iloc[-1] / px.iloc[0] - 1.0)
    years = (px.index[-1] - px.index[0]).days / 365.25
    years = max(years, 1e-9)
    cagr = (1.0 + total) ** (1.0 / years) - 1.0
    mdd = mdd_from_returns(ret)
    cash_daily = C.CASH_ANNUAL / 252
    sd = float(ret.std(ddof=1))
    sharpe = float((ret - cash_daily).mean() / sd * np.sqrt(252)) if sd > 0 else np.nan
    calmar = float(cagr / abs(mdd)) if mdd < 0 else np.nan
    return {"n_periods": len(ret), "cagr": cagr, "mdd": mdd, "sharpe": sharpe,
            "calmar": calmar, "trades": 0, "cost_total_pct": 0.0, "cost_ann_pct": 0.0,
            "total_return": total, "years": years, "n_rebal_invested": 0}
