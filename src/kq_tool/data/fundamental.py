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

    mcap_mil = latest.get("__market_cap_latest", latest.get(MCAP_FIN_KEY, 0)) or 0
    market_cap = float(mcap_mil) * 1e6
    net_income_avg_quarter = float(latest.get("당기순이익(천원)", 0) or 0) * 1000
    net_income_ttm = float(
        latest.get("__net_income_ttm", (latest.get("당기순이익(천원)", 0) or 0) * 4)
        or 0
    ) * 1000
    equity = float(latest.get("__equity_latest", latest.get("자본총계(천원)", 0)) or 0) * 1000
    shares = float(
        latest.get("__shares_latest", latest.get("기말발행주식수(보통주)(주)", 0)) or 0
    )

    pe_calc = market_cap / net_income_ttm if net_income_ttm > 0 else None
    pbr_calc = market_cap / equity if equity > 0 else None
    eps_calc = net_income_ttm / shares if shares > 0 else None
    roe_latest = latest.get("__roe_latest")
    roe_ttm_calc = latest.get("__roe_ttm_calc")
    if _valid_number(roe_latest, -200, 200):
        roe = float(roe_latest) / 100
        roe_basis = "latest_reported_quarter"
    elif _valid_number(roe_ttm_calc, -200, 200):
        roe = float(roe_ttm_calc) / 100
        roe_basis = "ttm_net_income_to_latest_equity"
    else:
        roe = float(latest.get("ROE(%)", 0) or 0) / 100
        roe_basis = "pit_snapshot"

    return {
        "trailingPE": metric_or_calc(pe_metric, pe_calc, 0, 200),
        "priceToBook": metric_or_calc(pbr_metric, pbr_calc, 0, 30),
        "returnOnEquity": round(roe, 4),
        "returnOnEquity_basis": roe_basis,
        "marketCap": market_cap if market_cap > 0 else None,
        "marketCap_basis": "pit_financial_snapshot",
        "fundamentalPeriodDate": latest.get("__latest_period_date"),
        "fundamentalObservableDate": latest.get("__latest_observable_date"),
        "fundamentalSource": "excel_pit",
        "beta": 1.0,
        "currentPrice": None,
        "trailingEps": round(float(eps_metric), 0)
        if eps_metric
        else (round(float(eps_calc), 0) if eps_calc else None),
        "freeCashflow": float(latest.get("영업활동으로인한현금흐름(천원)", 0) or 0) * 1000,
        "sharesOutstanding": shares,
        "equity": equity if equity > 0 else None,
        "netIncomeTTM": net_income_ttm if net_income_ttm > 0 else None,
        "netIncomeAvgQuarter": net_income_avg_quarter if net_income_avg_quarter > 0 else None,
        "bookValue": bps_metric,
        "dividendYield": (float(div_yield) / 100) if div_yield else None,
    }


def refresh_market_sensitive_fundamentals(
    info: Mapping[str, object] | None,
    current_price: float | None,
    current_date: str | None = None,
) -> dict:
    """Refresh price-sensitive fundamentals with the current analysis price.

    Accounting values such as ROE, equity, and net income remain tied to the
    latest point-in-time financial statement. Market cap, PER, and PBR are
    display/analysis fields that become stale when the old financial snapshot
    price is used for today's chart, so recompute them from the current price
    whenever share count is available.
    """

    out = dict(info or {})
    try:
        price = float(current_price) if current_price is not None else None
    except (TypeError, ValueError):
        price = None
    try:
        shares = float(out.get("sharesOutstanding") or 0)
    except (TypeError, ValueError):
        shares = 0.0
    try:
        equity = float(out.get("equity") or 0)
    except (TypeError, ValueError):
        equity = 0.0
    try:
        net_income_ttm = float(out.get("netIncomeTTM") or 0)
    except (TypeError, ValueError):
        net_income_ttm = 0.0

    out.setdefault("fundamentalSource", "unknown")
    if price is None or price <= 0 or shares <= 0:
        out["fundamental_note"] = (
            "PER/PBR/시총은 재무 스냅샷 기준입니다. 현재가 또는 주식수 부족으로 "
            "현재가 재계산을 적용하지 못했습니다."
        )
        return out

    market_cap = price * shares
    out["currentPrice"] = price
    out["marketCap"] = market_cap
    out["marketCap_basis"] = "current_price_x_shares"
    out["marketCap_date"] = current_date

    if equity > 0:
        out["priceToBook"] = round(market_cap / equity, 2)
        out["priceToBook_basis"] = "current_market_cap_to_pit_equity"
    if net_income_ttm > 0:
        out["trailingPE"] = round(market_cap / net_income_ttm, 2)
        out["trailingPE_basis"] = "current_market_cap_to_pit_ttm_net_income"

    out["fundamental_note"] = (
        "PER/PBR/시총은 현재가×주식수로 재계산했습니다. PER은 TTM 순이익, "
        "PBR은 최신 자본총계, ROE는 최신 PiT 재무제표 기준입니다."
    )
    return out
