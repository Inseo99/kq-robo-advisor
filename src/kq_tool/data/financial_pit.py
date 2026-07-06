"""Point-in-Time financial-data helpers.

The legacy FnGuide Excel cache stores fiscal period dates.  A fiscal quarter
ending on 2024-03-31 was not observable on that date, so historical screens and
backtests must shift financial rows to the date when they could reasonably have
been known.

Policy used here:
    - Q1/Q2/Q3 reports: quarter end + 45 calendar days
    - Q4/annual report: fiscal year end + 90 calendar days

The rule is intentionally conservative and deterministic.  It does not attempt
to recover exact DART filing timestamps, but it prevents the larger failure
mode: using a future financial row before publication.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence

import numpy as np
import pandas as pd

MCAP_KEY = "시가총액(티커-상장예정주식수 포함)(백만원)"

DEFAULT_FINANCIAL_ITEMS: tuple[str, ...] = (
    "매출액(천원)",
    "영업이익(천원)",
    "당기순이익(천원)",
    "ROE(%)",
    "ROA(%)",
    "자산총계(천원)",
    "부채총계(천원)",
    "자본총계(천원)",
    "영업활동으로인한현금흐름(천원)",
    MCAP_KEY,
    "기말발행주식수(보통주)(주)",
    "외국인지분율(%)",
)

_NORMALIZED_CACHE: dict[tuple[int, tuple[int, int]], pd.DataFrame] = {}
_MCAP_HISTORY_CACHE: dict[tuple[int, tuple[str, ...], tuple[str, ...] | None], pd.DataFrame] = {}


def _strip_tz(values: pd.Series) -> pd.Series:
    dates = pd.to_datetime(values, errors="coerce")
    try:
        return dates.dt.tz_localize(None)
    except TypeError:
        return dates


def _normalize_ticker(value: object) -> str:
    return str(value).strip().lstrip("A").zfill(6)


def financial_publication_lag_days(period_date: object) -> int:
    """Return conservative DART publication lag in calendar days."""

    date = pd.Timestamp(period_date)
    return 90 if int(date.quarter) == 4 else 45


def observable_date_for_period(period_date: object) -> pd.Timestamp:
    """Return the first date on which a financial period is considered usable."""

    date = pd.Timestamp(period_date).tz_localize(None)
    return date + pd.Timedelta(days=financial_publication_lag_days(date))


def normalize_financial_frame(fin_df: pd.DataFrame | None) -> pd.DataFrame:
    """Return long financial rows with period_date and observable_date columns."""

    columns = ["ticker", "item", "period_date", "observable_date", "value"]
    if fin_df is None or len(fin_df) == 0:
        return pd.DataFrame(columns=columns)

    cache_key = (id(fin_df), tuple(fin_df.shape))
    cached = _NORMALIZED_CACHE.get(cache_key)
    if cached is not None:
        return cached

    frame = fin_df.reset_index() if isinstance(fin_df.index, pd.MultiIndex) else fin_df.copy()
    if "아이템명" not in frame.columns and "item" in frame.columns:
        frame = frame.rename(columns={"item": "아이템명"})
    required = {"ticker", "아이템명", "date", "value"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"financial frame missing columns: {sorted(missing)}")

    out = frame[["ticker", "아이템명", "date", "value"]].copy()
    out["ticker"] = out["ticker"].map(_normalize_ticker)
    out["item"] = out["아이템명"].astype(str)
    out["period_date"] = _strip_tz(out["date"])
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    out = out.dropna(subset=["ticker", "item", "period_date", "value"])
    if out.empty:
        return pd.DataFrame(columns=columns)

    lag_days = np.where(out["period_date"].dt.quarter.eq(4), 90, 45)
    out["observable_date"] = out["period_date"] + pd.to_timedelta(lag_days, unit="D")
    result = out[columns].sort_values(["ticker", "item", "observable_date", "period_date"])
    _NORMALIZED_CACHE[cache_key] = result
    return result


def available_financial_rows(
    fin_df: pd.DataFrame | None,
    *,
    ticker: str | None = None,
    item: str | None = None,
    asof: object | None = None,
) -> pd.DataFrame:
    """Filter financial rows to what was observable at ``asof``."""

    rows = normalize_financial_frame(fin_df)
    if rows.empty:
        return rows
    if ticker is not None:
        rows = rows[rows["ticker"] == _normalize_ticker(ticker)]
    if item is not None:
        rows = rows[rows["item"] == item]
    cutoff = pd.Timestamp.today().normalize() if asof is None else pd.Timestamp(asof)
    cutoff = cutoff.tz_localize(None) if cutoff.tzinfo is not None else cutoff
    return rows[rows["observable_date"] <= cutoff].copy()


def latest_financials_asof(
    fin_df: pd.DataFrame | None,
    ticker: str,
    asof: object | None = None,
    *,
    items: Sequence[str] = DEFAULT_FINANCIAL_ITEMS,
    lookback_quarters: int = 4,
) -> dict[str, float]:
    """Return trailing financial values using only rows observable by ``asof``."""

    rows = available_financial_rows(fin_df, ticker=ticker, asof=asof)
    if rows.empty:
        return {}

    out: dict[str, float] = {}
    for item in items:
        sub = rows[rows["item"] == item].sort_values(["observable_date", "period_date"])
        if sub.empty:
            continue
        vals = pd.to_numeric(sub["value"], errors="coerce").dropna().tail(lookback_quarters)
        if not vals.empty:
            out[item] = float(vals.mean())
    return out


def financial_wide_monthly(
    fin_df: pd.DataFrame | None,
    item: str,
    *,
    tickers: Iterable[str] | None = None,
    start_date: object | None = None,
    end_date: object | None = None,
) -> pd.DataFrame:
    """Monthly wide panel where values appear only from observable_date onward."""

    rows = available_financial_rows(fin_df, item=item, asof=end_date)
    if tickers is not None:
        wanted = {_normalize_ticker(t) for t in tickers}
        rows = rows[rows["ticker"].isin(wanted)]
    if rows.empty:
        return pd.DataFrame()

    wide = rows.pivot_table(
        index="observable_date",
        columns="ticker",
        values="value",
        aggfunc="last",
    ).sort_index()
    wide = wide.resample("ME").last().ffill()
    if start_date is not None:
        wide = wide[wide.index >= pd.Timestamp(start_date)]
    if end_date is not None:
        wide = wide[wide.index <= pd.Timestamp(end_date)]
    return wide


def latest_mcap_tickers_asof(
    fin_df: pd.DataFrame | None,
    *,
    asof: object | None = None,
    top_n: int | None = None,
) -> list[str]:
    """Return tickers ranked by market cap observable at ``asof``."""

    rows = available_financial_rows(fin_df, item=MCAP_KEY, asof=asof)
    if rows.empty:
        return []
    latest = (
        rows.sort_values(["observable_date", "period_date"])
        .groupby("ticker")["value"]
        .last()
        .pipe(pd.to_numeric, errors="coerce")
        .dropna()
        .sort_values(ascending=False)
    )
    if top_n is not None and top_n > 0:
        latest = latest.head(top_n)
    return latest.index.tolist()


def build_mcap_history_pit(
    universe: dict[str, object],
    ticker_to_code: Callable[[str], str],
    fin_data: pd.DataFrame | None,
    fin_tickers_cache: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Build date x ticker market-cap history using observable dates."""

    if fin_data is None:
        return pd.DataFrame()
    allowed = set(fin_tickers_cache) if fin_tickers_cache is not None else None
    cache_key = (
        id(fin_data),
        tuple(universe.keys()),
        tuple(sorted(allowed)) if allowed is not None else None,
    )
    cached = _MCAP_HISTORY_CACHE.get(cache_key)
    if cached is not None:
        return cached

    rows = available_financial_rows(fin_data, item=MCAP_KEY, asof=None)
    if rows.empty:
        return pd.DataFrame()

    code_to_yahoo: dict[str, str] = {}
    for yahoo_ticker in universe:
        code = _normalize_ticker(ticker_to_code(yahoo_ticker))
        if allowed is not None and code not in allowed:
            continue
        code_to_yahoo[code] = yahoo_ticker

    rows = rows[rows["ticker"].isin(code_to_yahoo)]
    if rows.empty:
        return pd.DataFrame()

    series_by_ticker: dict[str, pd.Series] = {}
    for code, sub in rows.groupby("ticker", sort=False):
        yahoo_ticker = code_to_yahoo.get(code)
        if yahoo_ticker is None:
            continue
        values = (
            sub.sort_values(["observable_date", "period_date"])
            .drop_duplicates("observable_date", keep="last")
            .set_index("observable_date")["value"]
        )
        values = pd.to_numeric(values, errors="coerce").dropna()
        if not values.empty:
            series_by_ticker[yahoo_ticker] = values

    if not series_by_ticker:
        return pd.DataFrame()
    history = pd.DataFrame(series_by_ticker).apply(pd.to_numeric, errors="coerce")
    history = history.sort_index().ffill()
    _MCAP_HISTORY_CACHE[cache_key] = history
    return history
