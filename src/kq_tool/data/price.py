"""Price-data normalization helpers."""

from __future__ import annotations

import hashlib
from collections.abc import Callable

import numpy as np
import pandas as pd

PERIOD_DAYS: dict[str, int] = {
    "1d": 1,
    "1mo": 30,
    "3mo": 90,
    "6mo": 180,
    "1y": 365,
    "2y": 730,
    "3y": 1095,
    "5y": 1825,
    "7y": 2555,
    "10y": 3650,
    "12y": 4380,
    "15y": 5475,
    "20y": 7300,
}


def min_rows_for_period(period: str | None) -> int:
    """Return the minimum usable row count for a requested chart period."""

    return 1 if period == "1d" else 20


def days_for_period(period: str | None, default_days: int = 365) -> int:
    """Return calendar-day lookback for a UI/API period code."""

    return PERIOD_DAYS.get(period or "", default_days)


def normalize_yfinance_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Flatten yfinance MultiIndex columns into ordinary OHLCV columns."""

    if isinstance(frame.columns, pd.MultiIndex):
        frame = frame.copy()
        frame.columns = frame.columns.get_level_values(0)
    return frame


def extract_live_price(info: dict | None) -> float | None:
    """Extract a positive live price from a yfinance ``Ticker.info`` payload."""

    if not isinstance(info, dict):
        return None
    for key in ("currentPrice", "regularMarketPrice", "previousClose"):
        value = info.get(key)
        try:
            price = float(value)
        except (TypeError, ValueError):
            continue
        if price > 0 and np.isfinite(price):
            return price
    return None


def resolve_current_price_context(
    close_index: object,
    live_info: dict | None,
    *,
    now: object | None = None,
) -> dict[str, object]:
    """Resolve current-price override, source, and display date."""

    try:
        last_date = pd.Timestamp(close_index[-1]).strftime("%Y-%m-%d")  # type: ignore[index]
    except Exception:
        last_date = None

    live_price = extract_live_price(live_info)
    if live_price is not None and live_price > 0:
        return {
            "price": float(live_price),
            "source": "실시간",
            "date": pd.Timestamp(now if now is not None else pd.Timestamp.now()).strftime("%Y-%m-%d"),
        }
    return {"price": None, "source": "엑셀", "date": last_date}


def has_price_history(frame: object) -> bool:
    """Return whether a yfinance history result contains usable rows."""

    return frame is not None and not bool(getattr(frame, "empty", True))


def has_min_rows_for_period(frame: object, period: str | None) -> bool:
    """Return whether a price frame has enough rows for a UI/API period."""

    if not has_price_history(frame):
        return False
    try:
        return len(frame) >= min_rows_for_period(period)
    except Exception:
        return False


def prepare_yfinance_price_frame(frame: pd.DataFrame | None, period: str | None) -> pd.DataFrame | None:
    """Normalize and validate a yfinance download result for server use."""

    if not has_price_history(frame):
        return None
    normalized = normalize_yfinance_columns(frame)
    return normalized if has_min_rows_for_period(normalized, period) else None


def warm_yfinance_session(
    ticker: str = "005930.KS",
    period: str = "5d",
    *,
    history_loader: Callable[[str, str], object] | None = None,
) -> bool:
    """Check whether a yfinance session can return recent price history."""

    try:
        if history_loader is None:
            import yfinance as yf

            history_loader = lambda symbol, lookback: yf.Ticker(symbol).history(period=lookback)
        return has_price_history(history_loader(ticker, period))
    except Exception:
        return False


def filter_price_period(
    frame: pd.DataFrame,
    period: str | None,
    default_days: int = 365,
) -> pd.DataFrame:
    """Filter a price frame using the frame's last date as the anchor.

    The anchor must be the data's last available date, not today's date. This
    keeps Excel-backed historical data from being accidentally truncated when
    the local file is not current to today's calendar date.
    """

    if frame is None or frame.empty or not period or period == "max":
        return frame

    days = days_for_period(period, default_days)
    last_date = frame.index[-1]
    cutoff = last_date - pd.Timedelta(days=days)
    filtered = frame[frame.index >= cutoff]
    return filtered if len(filtered) >= min_rows_for_period(period) else frame


def stable_seed(value: str) -> int:
    """Build a process-stable NumPy seed from a string."""

    digest = hashlib.sha256(value.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little", signed=False)


def sample_price(
    ticker: str,
    n: int = 750,
    base: float = 50_000,
    end: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Create deterministic fallback OHLCV data for demos and offline mode."""

    rng = np.random.default_rng(stable_seed(ticker))
    prices = base * np.exp(np.cumsum(rng.normal(0.0002, 0.017, n)))
    index = pd.date_range(end=end or pd.Timestamp.now(), periods=n, freq="B")
    return pd.DataFrame(
        {
            "Open": prices * 0.99,
            "High": prices * 1.02,
            "Low": prices * 0.98,
            "Close": prices,
            "Volume": rng.integers(50_000, 3_000_000, n),
        },
        index=index,
    )
