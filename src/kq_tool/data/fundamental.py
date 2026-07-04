"""Fundamental-data helpers."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping

import numpy as np

MCAP_FIN_KEY = "시가총액(티커-상장예정주식수 포함)(백만원)"


def _stable_seed(value: str) -> int:
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little", signed=False)


def sample_fundamental_info(ticker: str) -> dict:
    """Create deterministic fallback fundamentals for demos and offline mode."""

    rng = np.random.default_rng(_stable_seed(ticker))
    price = float(rng.uniform(10_000, 120_000))
    eps = float(rng.uniform(500, 9_000))
    return {
        "trailingPE": round(price / eps, 1),
        "priceToBook": round(float(rng.uniform(0.5, 4.0)), 2),
        "returnOnEquity": round(float(rng.uniform(0.05, 0.28)), 3),
        "marketCap": round(price * float(rng.uniform(3e8, 6e9)), -8),
        "currentPrice": round(price, -2),
        "trailingEps": round(eps, 0),
    }


def has_yfinance_fundamental_info(info: object, required_key: str = "trailingPE") -> bool:
    """Return whether a yfinance info payload is usable by the legacy fallback."""

    return isinstance(info, Mapping) and bool(info.get(required_key))


def _valid_number(value: object, low: float, high: float) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return low < number < high


def metric_or_calc(
    metric_value: object,
    calc_value: object,
    low: float,
    high: float,
    digits: int = 2,
) -> float | None:
    """Prefer a direct metric, otherwise use a calculated fallback."""

    if _valid_number(metric_value, low, high):
        return float(metric_value)
    if _valid_number(calc_value, low, high):
        return round(float(calc_value), digits)
    return None


def excel_fundamental_info(
    latest: Mapping[str, float],
    metric_getter: Callable[[str], object],
) -> dict:
    """Build yfinance-style info dict from FnGuide-like latest financial rows."""

    pe_metric = metric_getter("per")
    pbr_metric = metric_getter("pbr")
    eps_metric = metric_getter("eps")
    bps_metric = metric_getter("bps")
    div_yield = metric_getter("div_yield")

    mcap_mil = latest.get(MCAP_FIN_KEY, 0) or 0
    market_cap = float(mcap_mil) * 1e6
    net_income = float(latest.get("당기순이익(천원)", 0) or 0) * 1000
    equity = float(latest.get("자본총계(천원)", 0) or 0) * 1000
    shares = float(latest.get("기말발행주식수(보통주)(주)", 0) or 0)

    pe_calc = market_cap / (net_income * 4) if net_income > 0 else None
    pbr_calc = market_cap / equity if equity > 0 else None
    eps_calc = net_income / shares if shares > 0 else None
    roe = float(latest.get("ROE(%)", 0) or 0) / 100

    return {
        "trailingPE": metric_or_calc(pe_metric, pe_calc, 0, 200),
        "priceToBook": metric_or_calc(pbr_metric, pbr_calc, 0, 30),
        "returnOnEquity": round(roe, 4),
        "marketCap": market_cap if market_cap > 0 else None,
        "beta": 1.0,
        "currentPrice": None,
        "trailingEps": round(float(eps_metric), 0)
        if eps_metric
        else (round(float(eps_calc), 0) if eps_calc else None),
        "freeCashflow": float(latest.get("영업활동으로인한현금흐름(천원)", 0) or 0) * 1000,
        "sharesOutstanding": shares,
        "bookValue": bps_metric,
        "dividendYield": (float(div_yield) / 100) if div_yield else None,
    }
