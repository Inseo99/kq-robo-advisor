"""Signal quality scoring helpers for adaptive technical signals."""

from __future__ import annotations

import numpy as np
import pandas as pd

from kq_tool.analyzer.indicators import bollinger_bands, macd, rsi
from kq_tool.config import SIGNAL_DIRECTION

SIGNAL_COLUMNS = list(SIGNAL_DIRECTION)


def _as_quantile(value: float) -> float:
    return float(value) / 100.0 if value > 1 else float(value)


def adaptive_rsi_thresholds(
    rsi_series: pd.Series,
    lookback: int = 120,
    low_pct: float = 15,
    high_pct: float = 85,
    min_periods: int = 30,
    low_cap: float = 35,
    high_floor: float = 65,
) -> tuple[pd.Series, pd.Series]:
    """Build point-in-time RSI thresholds from each stock's own RSI history."""

    clean = pd.to_numeric(rsi_series, errors="coerce")
    prior = clean.shift(1)
    low_q = _as_quantile(low_pct)
    high_q = _as_quantile(high_pct)
    min_periods = max(2, min(min_periods, lookback))

    low = prior.rolling(lookback, min_periods=min_periods).quantile(low_q)
    high = prior.rolling(lookback, min_periods=min_periods).quantile(high_q)
    low = low.combine_first(prior.expanding(min_periods=2).quantile(low_q))
    high = high.combine_first(prior.expanding(min_periods=2).quantile(high_q))

    return low.clip(upper=low_cap), high.clip(lower=high_floor)


def trend_strength(close: pd.Series, window: int = 20) -> pd.Series:
    """Return normalized rolling trend strength in the 0-1 range."""

    clean = pd.to_numeric(close, errors="coerce")

    def _score(values: np.ndarray) -> float:
        if np.isnan(values).any():
            return np.nan
        scale = float(np.std(values))
        if scale <= 1e-12:
            return 0.0
        x = np.arange(len(values), dtype=float)
        slope = float(np.polyfit(x, values, 1)[0])
        return float(max(0.0, min(1.0, abs(slope) / scale)))

    return clean.rolling(window, min_periods=window).apply(_score, raw=True)


def volume_confirmation(volume: pd.Series, window: int = 20, target_ratio: float = 1.5) -> pd.Series:
    """Score whether current volume confirms a signal versus prior average volume."""

    vol = pd.to_numeric(volume, errors="coerce").clip(lower=0)
    baseline = vol.shift(1).rolling(window, min_periods=max(3, min(window, 5))).mean()
    ratio = vol / (baseline + 1e-9)
    return (ratio / target_ratio).clip(lower=0.0, upper=1.5)


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """Compute On-Balance Volume from close and volume series."""

    direction = np.sign(pd.to_numeric(close, errors="coerce").diff()).fillna(0.0)
    vol = pd.to_numeric(volume, errors="coerce").fillna(0.0)
    return (direction * vol).cumsum()


def macd_signal_quality(
    macd_line: pd.Series,
    signal_line: pd.Series,
    slope_window: int = 3,
    scale_window: int = 26,
) -> tuple[pd.Series, pd.Series]:
    """Score MACD cross quality using zero-line position and MACD slope."""

    ml = pd.to_numeric(macd_line, errors="coerce")
    sl = pd.to_numeric(signal_line, errors="coerce")
    cross_up = (ml > sl) & (ml.shift(1) <= sl.shift(1))
    cross_down = (ml < sl) & (ml.shift(1) >= sl.shift(1))
    slope = ml - ml.shift(slope_window)
    min_periods = min(scale_window, max(2, slope_window + 1))
    scale = ml.rolling(scale_window, min_periods=min_periods).std()
    slope_score = (slope / (scale + 1e-9)).clip(lower=-1.0, upper=1.0)

    bull = 0.50 + np.where(ml > 0, 0.25, -0.10) + 0.25 * slope_score
    bear = 0.50 + np.where(ml < 0, 0.25, -0.10) - 0.25 * slope_score
    bull = pd.Series(bull, index=ml.index).where(cross_up, 0.0).clip(lower=0.0, upper=1.0)
    bear = pd.Series(bear, index=ml.index).where(cross_down, 0.0).clip(lower=0.0, upper=1.0)
    return bull, bear


def signal_streak_score(mask: pd.Series) -> pd.Series:
    """Score recent signal continuity: early confirmation is favored over stale streaks."""

    streaks = []
    streak = 0
    for flag in mask.fillna(False).astype(bool):
        streak = streak + 1 if flag else 0
        if streak == 0:
            streaks.append(0.0)
        elif streak <= 5:
            streaks.append(min(streak / 3.0, 1.0))
        else:
            streaks.append(max(0.3, 1.0 - (streak - 5) * 0.1))
    return pd.Series(streaks, index=mask.index)


def price_angle_score(close: pd.Series, window: int = 3, scale: float = 20.0) -> pd.Series:
    """Score short-term price movement magnitude in the 0-1 range."""

    clean = pd.to_numeric(close, errors="coerce")
    return (clean.pct_change(window).abs() * scale).clip(lower=0.0, upper=1.0)


def _quality_from_mask(
    mask: pd.Series,
    close: pd.Series,
    trend_weight: pd.Series,
    volume_weight: pd.Series,
) -> pd.Series:
    streak = signal_streak_score(mask)
    angle = price_angle_score(close).fillna(0.5)
    trend = pd.to_numeric(trend_weight, errors="coerce").fillna(0.7).clip(0.0, 1.0)
    volume = pd.to_numeric(volume_weight, errors="coerce").fillna(1.0).clip(0.4, 1.2)
    base = 0.45 * streak + 0.25 * angle + 0.30 * trend
    return (base * volume).where(mask.fillna(False), 0.0).clip(lower=0.0, upper=1.0)


def signal_quality_scores(
    close: pd.Series,
    volume: pd.Series | None = None,
    rsi_lookback: int = 120,
    trend_window: int = 20,
) -> pd.DataFrame:
    """Return 0-1 quality scores for the current RSI/MACD/BB signal family."""

    close = pd.to_numeric(close, errors="coerce")
    rsi14 = rsi(close)
    rsi_low, rsi_high = adaptive_rsi_thresholds(rsi14, lookback=rsi_lookback)
    upper, _, lower = bollinger_bands(close)
    macd_line, signal_line = macd(close)
    macd_bull, macd_bear = macd_signal_quality(macd_line, signal_line)

    trend = trend_strength(close, trend_window)
    mean_reversion_weight = (1.0 - 0.7 * trend).clip(lower=0.3, upper=1.0)
    if volume is None:
        vol_weight = pd.Series(1.0, index=close.index)
    else:
        vol_weight = volume_confirmation(volume).reindex(close.index)

    scores = pd.DataFrame(index=close.index)
    rsi_oversold = rsi14 <= rsi_low
    rsi_overbought = rsi14 >= rsi_high
    bb_lower_touch = close <= lower
    bb_upper_touch = close >= upper
    scores["RSI 과매도"] = _quality_from_mask(
        rsi_oversold, close, mean_reversion_weight, vol_weight
    )
    scores["RSI 과매수"] = _quality_from_mask(
        rsi_overbought, close, mean_reversion_weight, vol_weight
    )
    scores["MACD 골든크로스"] = (macd_bull * vol_weight.clip(0.4, 1.2)).clip(0.0, 1.0)
    scores["MACD 데드크로스"] = (macd_bear * vol_weight.clip(0.4, 1.2)).clip(0.0, 1.0)
    scores["BB 하단터치"] = _quality_from_mask(
        bb_lower_touch, close, mean_reversion_weight, vol_weight
    )
    scores["BB 상단터치"] = _quality_from_mask(
        bb_upper_touch, close, mean_reversion_weight, vol_weight
    )
    return scores[SIGNAL_COLUMNS].fillna(0.0).clip(lower=0.0, upper=1.0)


_adaptive_rsi_thresholds = adaptive_rsi_thresholds
_signal_quality_scores = signal_quality_scores
