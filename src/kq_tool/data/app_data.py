"""Application data gateway for UI/server wiring.

The app should display the same data that the validation pipeline uses. This
module is the single gateway for adjusted close prices, active price universe,
macro card values, and regime UI payloads.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
PRICE_PANEL = ROOT / "data" / "prices" / "close.csv"
MACRO_DIR = ROOT / "data" / "macro"


def ticker_to_code(ticker: str) -> str:
    """Convert Yahoo-style tickers to project price-panel codes."""

    value = str(ticker).strip()
    if value.startswith("^"):
        return value
    if "." in value:
        value = value.split(".", 1)[0]
    return value


@lru_cache(maxsize=1)
def load_close_panel(path: str | Path = PRICE_PANEL) -> pd.DataFrame:
    """Load the adjusted close panel used by validation/backtests."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, parse_dates=["date"]).set_index("date")
    frame.index = pd.to_datetime(frame.index).tz_localize(None)
    frame.columns = [str(col) for col in frame.columns]
    frame = frame.apply(pd.to_numeric, errors="coerce").sort_index()
    return frame


def latest_price_date() -> str | None:
    """Return the latest date with at least one adjusted close observation."""

    panel = load_close_panel()
    valid = panel.dropna(how="all")
    if valid.empty:
        return None
    return valid.index[-1].strftime("%Y-%m-%d")


def _period_days(period: str | None) -> int | None:
    if period in (None, "max", "all", "12y", "15y", "20y"):
        return None
    mapping = {"1d": 1, "5d": 5, "1m": 31, "3m": 93, "6m": 186, "1y": 365, "2y": 730, "3y": 1095, "5y": 1825, "7y": 2555}
    return mapping.get(str(period).lower())


def filter_period(series: pd.Series, period: str | None) -> pd.Series:
    series = series.dropna().sort_index()
    days = _period_days(period)
    if days is None or series.empty:
        return series
    cutoff = series.index[-1] - pd.Timedelta(days=days)
    filtered = series[series.index >= cutoff]
    return filtered if not filtered.empty else series


def _macro_series_as_close() -> pd.Series | None:
    path = MACRO_DIR / "kospi.csv"
    if not path.exists():
        return None
    frame = pd.read_csv(path, parse_dates=["date"]).set_index("date")
    values = pd.to_numeric(frame["value"], errors="coerce").dropna().sort_index()
    values.name = "Close"
    return values


def get_price_series(ticker: str, period: str | None = "max") -> pd.Series | None:
    """Return adjusted close series for a ticker or KOSPI benchmark."""

    code = ticker_to_code(ticker)
    if code in {"^KS11", "KOSPI", "kospi"}:
        series = _macro_series_as_close()
        return filter_period(series, period) if series is not None else None

    panel = load_close_panel()
    if code not in panel.columns:
        return None
    series = pd.to_numeric(panel[code], errors="coerce").dropna().sort_index()
    series.name = code
    return filter_period(series, period)


def get_price_frame(ticker: str, period: str | None = "max") -> pd.DataFrame | None:
    """Return an OHLCV-like frame built from adjusted close data."""

    close = get_price_series(ticker, period)
    if close is None or close.empty:
        return None
    frame = pd.DataFrame(index=close.index)
    frame["Close"] = close.astype(float)
    frame["Open"] = frame["Close"].shift(1).fillna(frame["Close"])
    frame["High"] = frame[["Open", "Close"]].max(axis=1)
    frame["Low"] = frame[["Open", "Close"]].min(axis=1)
    frame["Volume"] = 0
    return frame[["Open", "High", "Low", "Close", "Volume"]]


def get_universe(asof: str | pd.Timestamp | None = None, stale_days: int = 20) -> list[str]:
    """Return active price-panel codes as of the requested date.

    A code is active if its last available observation is within `stale_days` of
    the panel's as-of date. This removes stale/delisted names from app screens.
    """

    panel = load_close_panel()
    if asof is not None:
        cutoff = pd.Timestamp(asof)
        panel = panel.loc[:cutoff]
    panel = panel.dropna(how="all")
    if panel.empty:
        return []
    last_panel_date = panel.index[-1]
    min_date = last_panel_date - pd.Timedelta(days=stale_days)
    active: list[str] = []
    for code in panel.columns:
        series = panel[code].dropna()
        if not series.empty and series.index[-1] >= min_date:
            active.append(str(code))
    return active


def _read_macro_value(name: str) -> tuple[pd.Timestamp, float] | tuple[None, None]:
    path = MACRO_DIR / f"{name}.csv"
    if not path.exists():
        return None, None
    frame = pd.read_csv(path, parse_dates=["date"]).dropna()
    if frame.empty:
        return None, None
    row = frame.sort_values("date").iloc[-1]
    return pd.Timestamp(row["date"]), float(row["value"])


def _read_macro_change(name: str, periods: int = 21) -> float | None:
    path = MACRO_DIR / f"{name}.csv"
    if not path.exists():
        return None
    frame = pd.read_csv(path, parse_dates=["date"]).dropna().sort_values("date")
    values = pd.to_numeric(frame["value"], errors="coerce").dropna()
    if len(values) <= periods:
        return None
    previous = float(values.iloc[-periods - 1])
    current = float(values.iloc[-1])
    if previous == 0:
        return None
    return current / previous - 1.0


def get_macro_snapshot() -> dict[str, Any]:
    """Return latest observable macro values for the macro card.

    Missing values remain None; they are never silently converted to 0.00.
    """

    gdp_date, gdp = _read_macro_value("gdp_qoq")
    spread_date, spread = _read_macro_value("yield_spread_10y_3y")
    usd_date, usd = _read_macro_value("usdkrw")
    cpi_date, cpi = _read_macro_value("cpi_yoy")
    base_date, base = _read_macro_value("base_rate")
    usd_change = _read_macro_change("usdkrw", periods=21)

    indicators = {
        "GDP_QoQ": round(gdp, 3) if gdp is not None else None,
        "CPI_YoY": round(cpi, 3) if cpi is not None else None,
        "기준금리": round(base, 3) if base is not None else None,
        "장단기스프레드": round(spread, 3) if spread is not None else None,
        "환율USD": round(usd, 2) if usd is not None else None,
    }
    current_features = {
        "gdp_growth": (gdp / 100.0) if gdp is not None else None,
        "spread": spread,
        "usd_change": usd_change,
    }
    latest_dates = [d for d in [gdp_date, spread_date, usd_date, cpi_date, base_date] if d is not None]
    asof = max(latest_dates).strftime("%Y-%m-%d") if latest_dates else None
    return {
        "asof": asof,
        "indicators": indicators,
        "current_features": current_features,
        "current_hint": "data/macro CSV 관측가능 최신값 기준",
    }


def get_regime_payload() -> dict[str, Any]:
    """Return the unified regime UI payload."""

    from kq_tool.regime.ui_payload import build_payload

    return build_payload()


__all__ = [
    "filter_period",
    "get_macro_snapshot",
    "get_price_frame",
    "get_price_series",
    "get_regime_payload",
    "get_universe",
    "latest_price_date",
    "load_close_panel",
    "ticker_to_code",
]
