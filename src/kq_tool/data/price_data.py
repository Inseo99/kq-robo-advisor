"""Price data integrity helpers.

These functions are meant to guard the raw price panel used by screeners,
backtests, and strategy validation. The central heuristic is Korea's daily price
limit: after 2015-06-15, ordinary common-stock daily moves are capped at +/-30%;
before that the cap was +/-15%. A much larger close-to-close jump is therefore
usually a corporate action such as split, reverse split, capital reduction, or a
bad unadjusted data source.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = [
    "CorporateAction",
    "adjust_for_actions",
    "audit_price_panel",
    "detect_corporate_actions",
    "find_stale_series",
    "krx_daily_limit",
]

KRX_LIMIT_CHANGE_DATE = pd.Timestamp("2015-06-15")


@dataclass(frozen=True)
class CorporateAction:
    """Detected split-like price jump."""

    ticker: str | None
    date: pd.Timestamp
    ratio: float
    raw_return: float
    limit: float
    kind: str


def krx_daily_limit(date: pd.Timestamp | str) -> float:
    """Return Korea's approximate daily price limit for a decision date."""

    ts = pd.Timestamp(date)
    return 0.30 if ts >= KRX_LIMIT_CHANGE_DATE else 0.15


def _clean_series(price: pd.Series) -> pd.Series:
    clean = pd.to_numeric(price, errors="coerce").dropna()
    clean = clean[clean > 0]
    clean.index = pd.to_datetime(clean.index)
    return clean.sort_index()


def detect_corporate_actions(
    price: pd.Series,
    *,
    ticker: str | None = None,
    tolerance: float = 0.02,
    min_ratio: float = 1.5,
) -> list[CorporateAction]:
    """Detect close-to-close jumps too large to be ordinary market returns.

    ``ratio`` is always >= 1. For a split-like drop it is previous/current; for
    a reverse-split-like jump it is current/previous.
    """

    clean = _clean_series(price)
    if len(clean) < 2:
        return []

    returns = clean.pct_change().dropna()
    actions: list[CorporateAction] = []
    for date, ret in returns.items():
        limit = krx_daily_limit(date)
        if abs(float(ret)) <= limit + tolerance:
            continue
        prev = float(clean.loc[:date].iloc[-2])
        cur = float(clean.loc[date])
        if prev <= 0 or cur <= 0:
            continue
        ratio = max(prev / cur, cur / prev)
        if ratio < min_ratio:
            continue
        kind = "split_like" if cur < prev else "reverse_split_like"
        actions.append(
            CorporateAction(
                ticker=ticker,
                date=pd.Timestamp(date),
                ratio=float(ratio),
                raw_return=float(ret),
                limit=float(limit),
                kind=kind,
            )
        )
    return actions


def adjust_for_actions(price: pd.Series, actions: list[CorporateAction] | None = None) -> pd.Series:
    """Back-adjust a single unadjusted close series for detected actions.

    For an abnormal event on date ``t``, all earlier prices are multiplied by
    ``price[t] / price[t-1]``. This removes the impossible jump and preserves
    daily returns away from event dates.
    """

    raw = _clean_series(price)
    if raw.empty:
        return raw
    adjusted = raw.astype(float).copy()
    detected = actions if actions is not None else detect_corporate_actions(raw)

    for action in sorted(detected, key=lambda item: item.date):
        if action.date not in raw.index:
            continue
        pos = raw.index.get_loc(action.date)
        if isinstance(pos, slice) or pos == 0:
            continue
        prev = float(raw.iloc[int(pos) - 1])
        cur = float(raw.iloc[int(pos)])
        if prev <= 0 or cur <= 0:
            continue
        factor = cur / prev
        adjusted.loc[adjusted.index < action.date] *= factor
    adjusted.name = price.name
    return adjusted


def find_stale_series(
    prices: pd.DataFrame,
    *,
    max_stale_days: int = 90,
    min_history: int = 20,
) -> list[str]:
    """Return columns whose last valid observation is far before panel end."""

    if prices is None or prices.empty:
        return []
    frame = prices.copy()
    frame.index = pd.to_datetime(frame.index)
    panel_end = frame.index.max()
    stale: list[str] = []
    for col in frame.columns:
        series = pd.to_numeric(frame[col], errors="coerce").dropna()
        if len(series) < min_history:
            continue
        last_date = pd.Timestamp(series.index.max())
        if (panel_end - last_date).days > max_stale_days:
            stale.append(str(col))
    return stale


def audit_price_panel(
    prices: pd.DataFrame,
    *,
    max_stale_days: int = 90,
) -> pd.DataFrame:
    """Audit a wide close-price panel and return a long issue report."""

    rows: list[dict[str, object]] = []
    if prices is None or prices.empty:
        return pd.DataFrame(columns=["ticker", "issue", "date", "detail", "ratio", "raw_return", "limit"])

    frame = prices.copy()
    frame.index = pd.to_datetime(frame.index)
    for col in frame.columns:
        actions = detect_corporate_actions(frame[col], ticker=str(col))
        for action in actions:
            rows.append(
                {
                    "ticker": str(col),
                    "issue": "unadjusted_action",
                    "date": action.date.strftime("%Y-%m-%d"),
                    "detail": action.kind,
                    "ratio": action.ratio,
                    "raw_return": action.raw_return,
                    "limit": action.limit,
                }
            )

    for col in find_stale_series(frame, max_stale_days=max_stale_days):
        series = pd.to_numeric(frame[col], errors="coerce").dropna()
        rows.append(
            {
                "ticker": str(col),
                "issue": "stale_series",
                "date": pd.Timestamp(series.index.max()).strftime("%Y-%m-%d") if not series.empty else "",
                "detail": f"last observation is more than {max_stale_days} days before panel end",
                "ratio": np.nan,
                "raw_return": np.nan,
                "limit": np.nan,
            }
        )

    columns = ["ticker", "issue", "date", "detail", "ratio", "raw_return", "limit"]
    return pd.DataFrame(rows, columns=columns)
