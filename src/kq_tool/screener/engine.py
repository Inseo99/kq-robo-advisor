"""Pure screener calculation helpers."""

from __future__ import annotations

from collections.abc import Callable, Mapping

import pandas as pd

from kq_tool.analyzer.dcf import reverse_dcf_growth
from kq_tool.analyzer.indicators import macd, rsi
from kq_tool.config import ROBO_BUY_THRESHOLD, ROBO_SELL_THRESHOLD

DEFAULT_PRICE_KINDS = ("open", "high", "low", "close", "volume")
DEFAULT_METRICS = ("per", "pbr", "eps", "bps", "div_yield")


def prewarm_screener_cache(
    load_price_parquet: Callable[[str], object],
    price_sheets: Mapping[str, str],
    metric_sheets: Mapping[str, str],
    *,
    price_kinds: tuple[str, ...] = DEFAULT_PRICE_KINDS,
    metrics: tuple[str, ...] = DEFAULT_METRICS,
) -> dict[str, list[str]]:
    """Preload parquet groups needed by the screener worker pool."""

    loaded: list[str] = []
    failed: list[str] = []
    for mapping, keys in ((price_sheets, price_kinds), (metric_sheets, metrics)):
        for key in keys:
            fname = mapping.get(key)
            if not fname:
                continue
            try:
                load_price_parquet(fname)
                loaded.append(fname)
            except Exception:
                failed.append(fname)
    return {"loaded": loaded, "failed": failed}


def latest_price_date_from_groups(price_groups: Mapping[str, pd.Series]) -> str | None:
    """Return the latest available close-date string from cached price groups."""

    last_dates = []
    for series in price_groups.values():
        try:
            clean = series.dropna()
            if len(clean) > 0:
                last_dates.append(clean.index[-1])
        except Exception:
            continue
    if not last_dates:
        return None
    return pd.Timestamp(max(last_dates)).strftime("%Y-%m-%d")


def optional_float(value: object) -> float | None:
    """Convert value to float, mapping zero/invalid values to None."""

    try:
        numeric = float(value or 0)
    except (TypeError, ValueError):
        return None
    return numeric or None


def momentum_recent(close: pd.Series, lookback: int = 60) -> float | None:
    """Return recent price momentum for the operating screener."""

    if len(close) < lookback:
        return None
    base = float(close.iloc[-lookback])
    if base <= 0:
        return None
    return float(close.iloc[-1] / base - 1)


def momentum_12_1(close: pd.Series) -> float | None:
    """Return S2 12-1 month momentum: one-month-ago price vs twelve-month-ago price."""

    if len(close) < 252:
        return None
    base = float(close.iloc[-252])
    if base <= 0:
        return None
    return float(close.iloc[-21] / base - 1)


def quick_robo_from_close(close: pd.Series) -> tuple[float, str]:
    """Fast RSI/MACD-only robo score for the screener table."""

    rsi14 = rsi(close)
    macd_line, signal_line = macd(close)
    rsi_value = float(rsi14.iloc[-1]) if pd.notna(rsi14.iloc[-1]) else 50.0
    macd_bull = (
        float(macd_line.iloc[-1]) > float(signal_line.iloc[-1])
        if pd.notna(macd_line.iloc[-1])
        else False
    )
    raw = (1 if rsi_value < 30 else -1 if rsi_value > 70 else 0) * 20
    raw += (1 if macd_bull else -1) * 25
    score = round(max(0, min(100, (raw + 45) / 0.9)), 1)
    signal = "매수" if score >= ROBO_BUY_THRESHOLD else (
        "매도" if score <= ROBO_SELL_THRESHOLD else "관망"
    )
    return score, signal


def build_screener_record(
    ticker: str,
    name: str,
    close: pd.Series,
    info: Mapping[str, object],
    required_return: float,
) -> dict:
    """Build the screener payload for one ticker from already-loaded data."""

    pe = optional_float(info.get("trailingPE"))
    pbr = optional_float(info.get("priceToBook"))
    roe = optional_float(info.get("returnOnEquity"))
    eps = optional_float(info.get("trailingEps"))
    current = float(close.iloc[-1])
    change = float((close.iloc[-1] - close.iloc[-2]) / close.iloc[-2] * 100) if len(close) > 1 else 0.0
    dcf_growth = reverse_dcf_growth(current, eps, required_return)
    score, signal = quick_robo_from_close(close)
    momentum = momentum_recent(close)
    s2_momentum = momentum_12_1(close)

    return {
        "name": name,
        "pe": pe,
        "pbr": pbr,
        "roe": roe,
        "mom": momentum,
        "s2_mom": s2_momentum,
        "cur": round(current, 0),
        "chg": round(change, 2),
        "dcf_g": round(dcf_growth, 4) if dcf_growth is not None else None,
        "score": score,
        "signal": signal,
    }
