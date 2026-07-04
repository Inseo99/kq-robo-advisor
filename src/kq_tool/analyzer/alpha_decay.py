"""Alpha Decay utilities for signal validity estimation."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from kq_tool.analyzer.indicators import bollinger_bands, macd, rsi
from kq_tool.config import HORIZONS, SIGNAL_DIRECTION


def _exp_decay(t: np.ndarray, a: float, lam: float, c: float) -> np.ndarray:
    return a * np.exp(-lam * t) + c


def half_life_to_confidence(half_life: float | None, count: int) -> float:
    """Map half-life and sample count to a conservative 0-10 confidence score."""

    if half_life is None or half_life <= 0:
        return 2.0
    if half_life < 3:
        confidence = 5.0 + (half_life / 3) * 2.0
    elif half_life <= 12:
        confidence = 7.0 + (12 - half_life) / 9 * 3.0
    elif half_life <= 20:
        confidence = 7.0 - (half_life - 12) / 8 * 3.0
    else:
        confidence = max(2.0, 4.0 - (half_life - 20) / 10 * 2.0)
    if count < 10:
        confidence *= 0.85
    elif count < 20:
        confidence *= 0.95
    return round(min(10.0, max(0.0, confidence)), 1)


def alpha_single(
    close: pd.Series,
    horizons: list[int] | None = None,
    active_window: int = 10,
    limit_threshold: float = 0.295,
    quality_scores: pd.DataFrame | Mapping[str, pd.Series] | None = None,
    min_signal_quality: float | None = None,
    weight_by_quality: bool = False,
) -> dict[str, dict]:
    """Estimate signal-level Alpha Decay for a single price series.

    Korean-market safeguards:
      - Treat a signal as active for a short window after the event.
      - Exclude signal dates around limit-up/limit-down style moves.
      - Convert bearish signals into directional edge returns.

    Optional signal-quality integration:
      - ``quality_scores`` may provide 0-1 scores by signal name.
      - ``min_signal_quality`` uses the score itself as the event filter.
      - ``weight_by_quality`` averages horizon returns with quality weights.
    """

    if horizons is None:
        horizons = HORIZONS
    quality_frame = None
    if quality_scores is not None:
        quality_frame = pd.DataFrame(quality_scores).reindex(close.index)
    rsi14 = rsi(close)
    macd_line, signal_line = macd(close)
    upper, _, lower = bollinger_bands(close)
    signals = {
        "RSI 과매도": rsi14 < 30,
        "RSI 과매수": rsi14 > 70,
        "MACD 골든크로스": (macd_line > signal_line)
        & (macd_line.shift(1) <= signal_line.shift(1)),
        "MACD 데드크로스": (macd_line < signal_line)
        & (macd_line.shift(1) >= signal_line.shift(1)),
        "BB 하단터치": close <= lower,
        "BB 상단터치": close >= upper,
    }
    limit_hit = close.pct_change().abs() >= limit_threshold
    result: dict[str, dict] = {}
    for signal_name, mask in signals.items():
        base_mask = mask.fillna(False).astype(bool)
        quality = None
        if quality_frame is not None and signal_name in quality_frame:
            quality = (
                pd.to_numeric(quality_frame[signal_name], errors="coerce")
                .reindex(close.index)
                .fillna(0.0)
                .clip(lower=0.0, upper=1.0)
            )
        event_mask = base_mask
        if quality is not None and min_signal_quality is not None:
            event_mask = quality >= float(min_signal_quality)

        raw_dates = base_mask[base_mask].index
        event_dates = event_mask[event_mask].index
        active_slice = event_mask.iloc[-active_window:] if len(event_mask) else event_mask
        is_active = bool(active_slice.any()) if len(active_slice) else False
        active_age = None
        if is_active:
            active_locs = np.flatnonzero(event_mask.to_numpy(dtype=bool))
            if len(active_locs):
                active_age = int(len(event_mask) - 1 - active_locs[-1])
        direction = SIGNAL_DIRECTION.get(signal_name, (1, ""))[0]

        dates = []
        quality_values = []
        excluded_limit = 0
        for dt in event_dates:
            idx = close.index.get_loc(dt)
            if isinstance(idx, slice):
                idx = idx.stop - 1
            bad_limit = bool(limit_hit.iloc[idx])
            if idx + 1 < len(close):
                bad_limit = bad_limit or bool(limit_hit.iloc[idx + 1])
            if bad_limit:
                excluded_limit += 1
                continue
            dates.append(dt)
            if quality is not None:
                quality_values.append(float(quality.loc[dt]))

        if len(dates) < 5:
            result[signal_name] = {
                "count": int(len(dates)),
                "raw_count": int(len(raw_dates)),
                "quality_count": int(len(event_dates)),
                "quality_threshold": min_signal_quality,
                "avg_quality": round(float(np.mean(quality_values)), 3)
                if quality_values
                else None,
                "quality_weighted": bool(weight_by_quality and quality is not None),
                "excluded_limit": int(excluded_limit),
                "half_life": None,
                "horizon_rets": {},
                "fit_y": None,
                "is_active": is_active,
                "active_age": active_age,
                "active_window": active_window,
                "direction": direction,
                "status": "표본 부족",
            }
            continue

        horizon_returns = {}
        for horizon in horizons:
            returns = []
            weights = []
            for dt in dates:
                idx = close.index.get_loc(dt)
                if isinstance(idx, slice):
                    idx = idx.stop - 1
                if idx + horizon < len(close):
                    raw_ret = float((close.iloc[idx + horizon] - close.iloc[idx]) / close.iloc[idx])
                    returns.append(direction * raw_ret)
                    if quality is not None:
                        weights.append(max(float(quality.loc[dt]), 1e-6))
            if not returns:
                horizon_returns[horizon] = None
            elif weight_by_quality and quality is not None and weights:
                horizon_returns[horizon] = float(np.average(returns, weights=weights))
            else:
                horizon_returns[horizon] = float(np.mean(returns))

        valid = [(h, r) for h, r in horizon_returns.items() if r is not None]
        half_life = None
        fit_y = None
        if len(valid) >= 4:
            try:
                x = np.array([h for h, _ in valid], dtype=float)
                y = np.array([ret for _, ret in valid], dtype=float)
                params, _ = curve_fit(_exp_decay, x, y, p0=[y[0], 0.1, 0.0], maxfev=3000)
                if params[1] > 0:
                    raw_half_life = float(np.log(2) / params[1])
                    if 0 < raw_half_life <= max(horizons) * 1.5:
                        half_life = round(raw_half_life, 1)
                fit_y = [round(float(_exp_decay(t, *params)) * 100, 3) for t in range(1, 31)]
            except Exception:
                pass

        result[signal_name] = {
            "count": int(len(dates)),
            "raw_count": int(len(raw_dates)),
            "quality_count": int(len(event_dates)),
            "quality_threshold": min_signal_quality,
            "avg_quality": round(float(np.mean(quality_values)), 3) if quality_values else None,
            "quality_weighted": bool(weight_by_quality and quality is not None),
            "excluded_limit": int(excluded_limit),
            "half_life": half_life,
            "horizon_rets": {
                str(h): round(ret * 100, 3) if ret is not None else None
                for h, ret in horizon_returns.items()
            },
            "fit_y": fit_y,
            "is_active": is_active,
            "active_age": active_age,
            "active_window": active_window,
            "direction": direction,
            "status": "측정됨" if half_life is not None else "감쇠 불안정",
        }
    return result


# Temporary compatibility aliases while legacy server.py is migrated.
_alpha_single = alpha_single
_half_life_to_confidence = half_life_to_confidence
