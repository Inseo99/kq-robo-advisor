"""Stock-analysis orchestration payload builder."""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from kq_tool.analyzer.alpha_decay import alpha_single
from kq_tool.analyzer.chart import build_stock_chart
from kq_tool.analyzer.indicators import atr, bollinger_bands, close_series, macd, rsi
from kq_tool.analyzer.robo import confidence_weighted_robo, legacy_robo_score, score_to_signal
from kq_tool.data.fundamental import refresh_market_sensitive_fundamentals


def _rounded_price(value: float) -> float:
    value = float(value)
    return round(value, 2) if abs(value) < 1000 else round(value, -1)


def analyze_stock_payload(
    ticker: str,
    frame: pd.DataFrame,
    *,
    period: str = "1y",
    info: Mapping[str, object] | None = None,
    name: str | None = None,
    is_sample: bool = False,
    current_price: float | None = None,
    current_source: str = "엑셀",
    current_date: str | None = None,
) -> dict:
    """Build the stock-analysis response from already-loaded data.

    The legacy app still owns I/O concerns such as Excel/yfinance loading.
    This function owns the deterministic analysis payload so it can be tested
    and reused by future API routes.
    """

    close = pd.to_numeric(close_series(frame), errors="coerce").dropna()
    if close.empty:
        raise ValueError("stock analysis requires at least one close price")

    frame = frame.reindex(close.index)
    ma20 = close.rolling(20).mean()
    ma60 = close.rolling(60).mean()
    ma200 = close.rolling(200).mean()
    rsi14 = rsi(close)
    macd_line, signal_line = macd(close)
    bb_upper, bb_middle, bb_lower = bollinger_bands(close)
    atr_series = atr(frame)

    chart = build_stock_chart(
        frame,
        close,
        indicators={
            "ma20": ma20,
            "ma60": ma60,
            "ma200": ma200,
            "bb_up": bb_upper,
            "bb_mid": bb_middle,
            "bb_lo": bb_lower,
            "rsi": rsi14,
            "macd": macd_line,
            "signal": signal_line,
        },
        period=period,
    )

    previous = float(close.iloc[-2]) if len(close) > 1 else float(close.iloc[-1])
    current = float(current_price) if current_price and current_price > 0 else float(close.iloc[-1])
    change_pct = (current - previous) / previous * 100 if previous else 0.0
    if current_date is None:
        current_date = close.index[-1].strftime("%Y-%m-%d")

    rsi_value = float(rsi14.iloc[-1]) if pd.notna(rsi14.iloc[-1]) else 50.0
    macd_value = float(macd_line.iloc[-1]) if pd.notna(macd_line.iloc[-1]) else 0.0
    signal_value = float(signal_line.iloc[-1]) if pd.notna(signal_line.iloc[-1]) else 0.0
    bb_upper_value = float(bb_upper.iloc[-1]) if pd.notna(bb_upper.iloc[-1]) else current * 1.05
    bb_lower_value = float(bb_lower.iloc[-1]) if pd.notna(bb_lower.iloc[-1]) else current * 0.95
    ma20_value = float(ma20.iloc[-1]) if pd.notna(ma20.iloc[-1]) else current
    ma60_value = float(ma60.iloc[-1]) if pd.notna(ma60.iloc[-1]) else current
    ma200_value = float(ma200.iloc[-1]) if pd.notna(ma200.iloc[-1]) else current
    atr_value = float(atr_series.iloc[-1]) if pd.notna(atr_series.iloc[-1]) else current * 0.02

    entry = current
    buy_low = max(0.0, entry - 0.25 * atr_value)
    buy_high = entry + 0.25 * atr_value
    target = entry + 2.0 * atr_value
    stop = max(0.0, entry - 1.5 * atr_value)

    alpha = alpha_single(close, active_window=10)

    s_rsi = 1 if rsi_value < 30 else (-1 if rsi_value > 70 else 0)
    s_macd = 1 if macd_value > signal_value else -1
    s_bb = 1 if current < bb_lower_value else (-1 if current > bb_upper_value else 0)
    s_ma20 = 1 if current > ma20_value else -1
    s_ma60 = 1 if current > ma60_value else -1

    score = legacy_robo_score(s_rsi, s_macd, s_bb, s_ma20, s_ma60)
    robo_signal = score_to_signal(score)
    confidence_weighted = confidence_weighted_robo(alpha, s_rsi, s_macd, s_bb, s_ma20, s_ma60)

    info = refresh_market_sensitive_fundamentals(info or {}, current, current_date)
    return {
        "ticker": ticker,
        "name": name or ticker,
        "is_sample": bool(is_sample),
        "cur": round(current, 0),
        "chg": round(change_pct, 2),
        "cur_source": current_source,
        "cur_date": current_date,
        "chart": chart,
        "indicators": {
            "rsi": round(rsi_value, 1),
            "macd": round(macd_value, 2),
            "signal": round(signal_value, 2),
            "bb_upper": round(bb_upper_value, 0),
            "bb_lower": round(bb_lower_value, 0),
            "ma20": round(ma20_value, 0),
            "ma60": round(ma60_value, 0),
            "ma200": round(ma200_value, 0),
            "atr": round(atr_value, 0),
        },
        "robo": {
            "score": score,
            "signal": robo_signal,
            "s_rsi": s_rsi,
            "s_macd": s_macd,
            "s_bb": s_bb,
            "s_ma20": s_ma20,
            "s_ma60": s_ma60,
            "entry": _rounded_price(entry),
            "buy_low": _rounded_price(buy_low),
            "buy_high": _rounded_price(buy_high),
            "target": _rounded_price(target),
            "stoploss": _rounded_price(stop),
            "rr": round(2 / 1.5, 2),
            "cw_score": confidence_weighted["score"],
            "cw_signal": confidence_weighted["signal"],
            "confidence": confidence_weighted["confidence"],
            "exit_days": confidence_weighted["exit_days"],
            "decay_state": confidence_weighted.get("decay_state"),
            "valid_days": confidence_weighted.get("valid_days"),
            "validity_basis": confidence_weighted.get("validity_basis"),
            "validity_text": confidence_weighted.get("validity_text"),
            "cw_detail": confidence_weighted["detail"],
        },
        "alpha": alpha,
        "fund": {
            "pe": info.get("trailingPE"),
            "pbr": info.get("priceToBook"),
            "roe": info.get("returnOnEquity"),
            "mcap": info.get("marketCap"),
            "source": info.get("fundamentalSource"),
            "note": info.get("fundamental_note"),
            "marketCap_basis": info.get("marketCap_basis"),
            "marketCap_date": info.get("marketCap_date"),
            "pe_basis": info.get("trailingPE_basis"),
            "pbr_basis": info.get("priceToBook_basis"),
            "roe_basis": info.get("returnOnEquity_basis"),
            "fundamental_period_date": info.get("fundamentalPeriodDate"),
            "fundamental_observable_date": info.get("fundamentalObservableDate"),
        },
    }


_analyze_stock_payload = analyze_stock_payload
