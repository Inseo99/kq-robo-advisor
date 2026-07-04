"""Point-in-Time market-cap helpers.

These functions are intentionally stateless. The caller owns caches and data
loading, while this module owns the transformation rules that prevent
look-ahead bias.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping

import pandas as pd

MCAP_KEY = "시가총액(티커-상장예정주식수 포함)(백만원)"


def build_top_marketcap_tickers(
    universe: Mapping[str, object],
    ticker_to_code: Callable[[str], str],
    fin_data: pd.DataFrame | None,
    fin_tickers_cache: Iterable[str] | None = None,
    limit: int | None = None,
    mcap_key: str = MCAP_KEY,
) -> list[str]:
    """Return tickers sorted by the latest available market cap.

    This helper matches the legacy screener behavior. It uses the latest
    available financial row, so it is suitable for today's screener universe,
    not historical backtests. Historical backtests should use
    ``get_top_mcap_at``.
    """

    tickers = list(universe.keys())
    if fin_data is None:
        return tickers[:limit] if limit is not None else tickers

    allowed = set(fin_tickers_cache) if fin_tickers_cache is not None else None
    pairs: list[tuple[str, float]] = []
    for ticker in tickers:
        code = ticker_to_code(ticker)
        try:
            if allowed is not None and code not in allowed:
                continue
            sub = fin_data.loc[code]
            if mcap_key not in sub.index:
                continue
            rows = sub.loc[mcap_key]
            if isinstance(rows, pd.Series):
                rows = rows.to_frame().T
            values = pd.to_numeric(rows["value"], errors="coerce").dropna()
            if not values.empty:
                pairs.append((ticker, float(values.iloc[-1])))
        except Exception:
            continue

    pairs.sort(key=lambda item: item[1], reverse=True)
    result = [ticker for ticker, _ in pairs]
    return result[:limit] if limit is not None else result


def build_mcap_history(
    universe: Mapping[str, object],
    ticker_to_code: Callable[[str], str],
    fin_data: pd.DataFrame | None,
    fin_tickers_cache: Iterable[str] | None = None,
    mcap_key: str = MCAP_KEY,
) -> pd.DataFrame:
    """Build a date x ticker market-cap DataFrame from financial rows."""

    if fin_data is None:
        return pd.DataFrame()

    allowed = set(fin_tickers_cache) if fin_tickers_cache is not None else None
    series_by_ticker: dict[str, pd.Series] = {}

    for ticker in universe:
        code = ticker_to_code(ticker)
        try:
            if allowed is not None and code not in allowed:
                continue
            sub = fin_data.loc[code]
            if mcap_key not in sub.index:
                continue

            rows = sub.loc[mcap_key]
            if isinstance(rows, pd.Series):
                rows = rows.to_frame().T

            frame = rows[["date", "value"]].dropna()
            if frame.empty:
                continue

            values = frame.set_index(pd.to_datetime(frame["date"]))["value"]
            values = pd.to_numeric(values, errors="coerce").dropna()
            if not values.empty:
                series_by_ticker[ticker] = values
        except Exception:
            continue

    if not series_by_ticker:
        return pd.DataFrame()

    history = pd.DataFrame(series_by_ticker)
    history = history.apply(pd.to_numeric, errors="coerce")
    return history.sort_index().ffill()


def get_top_mcap_at(mcap_history: pd.DataFrame | None, date: object, n: int = 200) -> list[str]:
    """Return top ``n`` tickers using only market-cap data known by ``date``."""

    if mcap_history is None or mcap_history.empty:
        return []

    as_of = pd.Timestamp(date)
    available = mcap_history.index[mcap_history.index <= as_of]
    if len(available) == 0:
        return []

    valid = pd.to_numeric(mcap_history.loc[available[-1]], errors="coerce").dropna()
    if valid.empty:
        return []

    return valid.nlargest(min(n, len(valid))).index.tolist()
