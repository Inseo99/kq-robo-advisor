"""Point-in-Time market-cap helpers.

These functions are intentionally stateless. The caller owns caches and data
loading, while this module owns the transformation rules that prevent
look-ahead bias.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping

import pandas as pd

from .financial_pit import build_mcap_history_pit, latest_mcap_tickers_asof

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

    # Today's screener can use the latest observable market cap, but it must
    # still respect publication lag.  Rank by code first, then map back to the
    # Yahoo-style ticker used by the app.
    code_to_ticker = {ticker_to_code(ticker): ticker for ticker in tickers}
    result_codes = latest_mcap_tickers_asof(fin_data, top_n=None)
    result = [code_to_ticker[code] for code in result_codes if code in code_to_ticker]
    return result[:limit] if limit is not None else result


def build_mcap_history(
    universe: Mapping[str, object],
    ticker_to_code: Callable[[str], str],
    fin_data: pd.DataFrame | None,
    fin_tickers_cache: Iterable[str] | None = None,
    mcap_key: str = MCAP_KEY,
) -> pd.DataFrame:
    """Build a date x ticker market-cap DataFrame from financial rows."""

    return build_mcap_history_pit(universe, ticker_to_code, fin_data, fin_tickers_cache)


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
