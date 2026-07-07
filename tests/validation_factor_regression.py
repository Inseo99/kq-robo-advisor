"""
validation_factor_regression.py

Fama-French 3/5 factor regression for the project's ETF allocation portfolios.

Outputs:
  - tests/factor_regression_results.csv
  - tests/factor_returns_korea.csv

Notes:
  - MKT is proxied by KODEX 200 monthly excess return.
  - SMB/HML/RMW/CMA are built from cached Korean stock price/fundamental data.
  - Characteristics are shifted by one month before forming factor spreads.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TESTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(TESTS_DIR))
sys.path.insert(0, str(ROOT / "src"))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from kq_tool.validation.factor_analysis import (  # noqa: E402
    cross_sectional_factor_returns,
    factor_regression_rows,
    run_factor_regressions,
)
from kq_tool.data.financial_pit import (  # noqa: E402
    financial_wide_monthly,
    latest_mcap_tickers_asof,
)
from validation_asset_allocation_v2 import (  # noqa: E402
    load_etf_data,
    run_phase1_all_portfolios,
)


MCAP_KEY = "시가총액(티커-상장예정주식수 포함)(백만원)"
ROA_KEY = "ROA(%)"
ROE_KEY = "ROE(%)"
ASSET_KEY = "자산총계(천원)"
KODEX200 = "069500.KS"


def _ticker_norm(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lstrip("A").str.zfill(6)


def _latest_mcap_tickers(
    fin: pd.DataFrame,
    top_n: int | None,
    *,
    asof: str | None = None,
) -> list[str]:
    return latest_mcap_tickers_asof(fin, asof=asof, top_n=top_n)


def _monthly_wide_from_long(
    path: Path,
    *,
    tickers: list[str],
    transform: str | None = None,
) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    frame["ticker"] = _ticker_norm(frame["ticker"])
    frame = frame[frame["ticker"].isin(tickers)]
    frame["date"] = pd.to_datetime(frame["date"])
    values = pd.to_numeric(frame["value"], errors="coerce")
    if transform == "inverse":
        values = np.where(values > 0, 1.0 / values, np.nan)
    frame = frame.assign(value=values).dropna(subset=["value"])
    wide = frame.pivot_table(index="date", columns="ticker", values="value", aggfunc="last")
    return wide.resample("ME").last().sort_index()


def _monthly_fin_item(fin: pd.DataFrame, item: str, *, tickers: list[str]) -> pd.DataFrame:
    return financial_wide_monthly(fin, item, tickers=tickers).sort_index()


def build_korea_ff_factors(
    *,
    top_n: int | None = 500,
    risk_free_rate: float = 0.02,
    quantile: float = 0.3,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Build monthly Korean FF-style factors from cached stock data."""

    cache = ROOT / "data" / "cache"
    fin_path = cache / "fin.parquet"
    close_path = cache / "price_수정종가.parquet"
    pbr_path = cache / "metric_PBR.parquet"

    fin = pd.read_parquet(fin_path)
    fin["date"] = pd.to_datetime(fin["date"])
    tickers = _latest_mcap_tickers(fin, top_n, asof=end_date)
    if not tickers:
        raise RuntimeError("No market-cap tickers available for factor construction.")

    close_m = _monthly_wide_from_long(close_path, tickers=tickers)
    returns = close_m.pct_change().replace([np.inf, -np.inf], np.nan)

    btm = _monthly_wide_from_long(pbr_path, tickers=tickers, transform="inverse")
    size = _monthly_fin_item(fin, MCAP_KEY, tickers=tickers)
    profitability = _monthly_fin_item(fin, ROA_KEY, tickers=tickers)
    if profitability.dropna(how="all").empty:
        profitability = _monthly_fin_item(fin, ROE_KEY, tickers=tickers)
    assets = _monthly_fin_item(fin, ASSET_KEY, tickers=tickers)
    investment = assets.pct_change(12)

    etf_df = load_etf_data()
    market_return = etf_df[KODEX200].resample("ME").last().pct_change()

    common_start = max(
        dt
        for dt in [
            returns.dropna(how="all").index.min(),
            btm.dropna(how="all").index.min(),
            size.dropna(how="all").index.min(),
            market_return.dropna().index.min(),
        ]
        if pd.notna(dt)
    )
    returns = returns[returns.index >= common_start]

    factors = cross_sectional_factor_returns(
        returns,
        market_return=market_return,
        risk_free_rate=risk_free_rate,
        size=size.shift(1),
        book_to_market=btm.shift(1),
        profitability=profitability.shift(1),
        investment=investment.shift(1),
        quantile=quantile,
    )
    factors = factors.replace([np.inf, -np.inf], np.nan).dropna(how="any")
    if start_date:
        factors = factors[factors.index >= pd.Timestamp(start_date)]
    if end_date:
        factors = factors[factors.index <= pd.Timestamp(end_date)]
    return factors


def build_portfolio_returns(
    *,
    start_date: str | None = "2014-06-26",
    end_date: str | None = None,
) -> pd.DataFrame:
    """Run the existing 10 allocation backtests and return monthly returns."""

    etf_df = load_etf_data()
    bt_start = max(etf_df.index[0], pd.Timestamp(start_date)) if start_date else etf_df.index[0]
    bt_end = pd.Timestamp(end_date) if end_date else etf_df.index[-1]
    results = run_phase1_all_portfolios(etf_df, start_date=bt_start, end_date=bt_end)

    series_by_name: dict[str, pd.Series] = {}
    for name, result in results.items():
        if "error" in result:
            continue
        equity = result.get("equity")
        if isinstance(equity, pd.Series):
            returns = equity.pct_change().dropna()
        else:
            dates = [item.get("date") for item in result.get("weights_log", [])]
            values = result.get("monthly_rets", [])
            returns = pd.Series(values, index=pd.to_datetime(dates)).dropna()
        if not returns.empty:
            series_by_name[name] = returns.astype(float)
    return pd.DataFrame(series_by_name).sort_index()


def run_validation(
    *,
    model: str = "ff5",
    top_n: int | None = 500,
    risk_free_rate: float = 0.02,
    start_date: str | None = "2014-06-26",
    end_date: str | None = None,
    quantile: float = 0.3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    print("=" * 72)
    print(f"KQ Quant Tool - Fama-French {model.upper()} factor regression")
    print("=" * 72)
    print(f"factor universe: {'all' if not top_n else f'top {top_n} by latest market cap'}")
    print(f"risk-free rate: {risk_free_rate:.2%} annual")

    t0 = time.time()
    print("\n[1] Build Korean FF-style factors")
    factors = build_korea_ff_factors(
        top_n=top_n,
        risk_free_rate=risk_free_rate,
        quantile=quantile,
        start_date=start_date,
        end_date=end_date,
    )
    if factors.empty:
        raise RuntimeError("Factor return frame is empty.")
    print(f"  factors: {list(factors.columns)}")
    print(f"  factor period: {factors.index[0].date()} ~ {factors.index[-1].date()} ({len(factors)} months)")

    print("\n[2] Build portfolio monthly returns")
    portfolios = build_portfolio_returns(start_date=start_date, end_date=end_date)
    common = portfolios.index.intersection(factors.index)
    portfolios = portfolios.loc[common].dropna(how="all")
    factors = factors.loc[common]
    print(f"  portfolios: {len(portfolios.columns)}")
    print(f"  regression months: {len(common)}")

    print("\n[3] Run factor regressions")
    results = run_factor_regressions(
        portfolios,
        factors,
        model=model,
        risk_free_rate=risk_free_rate,
        mkt_is_excess=True,
    )
    rows = pd.DataFrame(factor_regression_rows(results))
    rows = rows.sort_values(["alpha_t", "alpha_annual_pct"], ascending=False, na_position="last")

    out_csv = TESTS_DIR / "factor_regression_results.csv"
    factor_csv = TESTS_DIR / "factor_returns_korea.csv"
    rows.to_csv(out_csv, index=False, encoding="utf-8-sig")
    factors.to_csv(factor_csv, encoding="utf-8-sig")

    print("\n" + "-" * 92)
    print(f"{'Portfolio':<24} {'Alpha(ann)':>11} {'t(alpha)':>10} {'R2':>8} {'MKT beta':>10}")
    print("-" * 92)
    for _, row in rows.iterrows():
        alpha = row.get("alpha_annual_pct")
        alpha_t = row.get("alpha_t")
        r2 = row.get("r2")
        beta_mkt = row.get("beta_MKT")
        print(
            f"{str(row['portfolio'])[:24]:<24} "
            f"{float(alpha):>10.2f}% "
            f"{float(alpha_t):>10.2f} "
            f"{float(r2):>8.3f} "
            f"{float(beta_mkt):>10.3f}"
        )
    print("-" * 92)
    print(f"\nSaved: {out_csv}")
    print(f"Saved: {factor_csv}")
    print(f"Elapsed: {time.time() - t0:.1f}s")
    return rows, factors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["ff3", "ff5"], default="ff5")
    parser.add_argument("--top-n", type=int, default=500, help="0 means all available tickers")
    parser.add_argument("--risk-free", type=float, default=0.02, help="annual risk-free rate")
    parser.add_argument("--start", default="2014-06-26")
    parser.add_argument("--end", default=None)
    parser.add_argument("--quantile", type=float, default=0.3)
    args = parser.parse_args()

    run_validation(
        model=args.model,
        top_n=None if args.top_n <= 0 else args.top_n,
        risk_free_rate=args.risk_free,
        start_date=args.start,
        end_date=args.end,
        quantile=args.quantile,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
