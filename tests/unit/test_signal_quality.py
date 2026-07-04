from __future__ import annotations

import numpy as np
import pandas as pd

from kq_tool.analyzer.signal_quality import (
    SIGNAL_COLUMNS,
    adaptive_rsi_thresholds,
    macd_signal_quality,
    obv,
    signal_quality_scores,
    trend_strength,
    volume_confirmation,
)
from kq_tool.config import SIGNAL_DIRECTION


def _sample_close(n: int = 180) -> pd.Series:
    index = pd.date_range("2023-01-02", periods=n, freq="B")
    trend = np.linspace(100, 130, n)
    cycle = np.sin(np.arange(n) / 4) * 7
    return pd.Series(trend + cycle, index=index, name="Close")


def test_adaptive_rsi_thresholds_use_prior_window_only() -> None:
    rsi_values = pd.Series(
        [40.0, 42.0, 44.0, 46.0, 48.0, 10.0],
        index=pd.date_range("2024-01-02", periods=6, freq="B"),
    )

    low, high = adaptive_rsi_thresholds(
        rsi_values, lookback=5, low_pct=20, high_pct=80, min_periods=2
    )

    prior = rsi_values.iloc[:-1]
    assert low.iloc[-1] == min(float(prior.quantile(0.20)), 35.0)
    assert high.iloc[-1] == max(float(prior.quantile(0.80)), 65.0)


def test_volume_confirmation_rewards_volume_spike() -> None:
    index = pd.date_range("2024-01-02", periods=8, freq="B")
    volume = pd.Series([100, 100, 100, 100, 100, 100, 100, 300], index=index)

    score = volume_confirmation(volume, window=5)

    assert score.iloc[-1] > score.iloc[-2]
    assert score.iloc[-1] > 1.0


def test_macd_signal_quality_scores_crosses_in_zero_line_context() -> None:
    index = pd.date_range("2024-01-02", periods=8, freq="B")
    macd_line = pd.Series([-0.4, -0.3, -0.2, -0.1, 0.2, 0.5, 0.8, 1.1], index=index)
    signal_line = pd.Series([-0.3, -0.2, -0.1, 0.0, 0.1, 0.4, 0.7, 1.0], index=index)

    bull, bear = macd_signal_quality(macd_line, signal_line, slope_window=2, scale_window=4)

    assert bull.max() > 0
    assert bear.max() == 0
    assert bull.max() <= 1


def test_trend_strength_and_obv_are_directional() -> None:
    index = pd.date_range("2024-01-02", periods=30, freq="B")
    flat = pd.Series(np.repeat(100.0, 30), index=index)
    rising = pd.Series(np.linspace(100, 130, 30), index=index)
    volume = pd.Series(np.repeat(100.0, 30), index=index)

    assert trend_strength(rising, window=20).iloc[-1] > trend_strength(flat, window=20).iloc[-1]
    assert obv(rising, volume).iloc[-1] > 0


def test_signal_quality_scores_are_bounded_and_named() -> None:
    close = _sample_close()
    volume = pd.Series(1000 + np.arange(len(close)) * 5, index=close.index)

    scores = signal_quality_scores(close, volume=volume, rsi_lookback=40)

    assert list(scores.columns) == list(SIGNAL_DIRECTION)
    assert (scores.min().min() >= 0.0) and (scores.max().max() <= 1.0)
    assert scores.to_numpy().sum() > 0


def test_signal_quality_columns_follow_config_signal_direction() -> None:
    assert SIGNAL_COLUMNS == list(SIGNAL_DIRECTION)
