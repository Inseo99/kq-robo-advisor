"""Regime-segmented Alpha Decay validation helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from kq_tool.validation.costs import apply_signal_costs

Event = tuple[str, pd.Timestamp, int, int, float]
DEFAULT_REGIMES = ("골디락스", "리플레이션", "스태그플레이션", "디플레이션")
UNKNOWN_REGIME = "미분류"


def normalize_regime_series(regime_series: pd.Series | None) -> pd.Series:
    """Return a sorted datetime-indexed regime series with empty values removed."""

    if regime_series is None:
        return pd.Series(dtype=object)
    series = regime_series.copy()
    if series.empty:
        return pd.Series(dtype=object)
    series.index = pd.to_datetime(series.index)
    series = series.sort_index().dropna()
    return series.astype(object)


def regime_at_date(regime_series: pd.Series | None, date: pd.Timestamp, fallback: str = UNKNOWN_REGIME) -> str:
    """Find the latest known regime at or before ``date``."""

    series = normalize_regime_series(regime_series)
    if series.empty:
        return fallback
    dt = pd.Timestamp(date)
    pos = series.index.searchsorted(dt, side="right") - 1
    if pos < 0:
        return fallback
    value = series.iloc[pos]
    if value is None or pd.isna(value):
        return fallback
    return str(value)


def split_events_by_regime(
    events_by_signal: Mapping[str, Sequence[Event]],
    regime_series: pd.Series | None,
    *,
    fallback: str = UNKNOWN_REGIME,
) -> dict[str, dict[str, list[Event]]]:
    """Group Alpha Decay events by point-in-time regime label."""

    grouped: dict[str, dict[str, list[Event]]] = {}
    for signal_name, events in events_by_signal.items():
        for event in events:
            regime = regime_at_date(regime_series, event[1], fallback=fallback)
            grouped.setdefault(regime, {}).setdefault(signal_name, []).append(event)
    return grouped


def _edge_return(close: pd.Series, index: int, horizon: int, direction: int) -> float | None:
    if index + horizon >= len(close):
        return None
    p0 = float(close.iloc[index])
    p1 = float(close.iloc[index + horizon])
    if p0 <= 0 or not np.isfinite(p0) or not np.isfinite(p1):
        return None
    return int(direction) * ((p1 - p0) / p0)


def random_return_pool_by_regime(
    price_map: Mapping[str, pd.DataFrame],
    regime_series: pd.Series | None,
    *,
    target_regime: str,
    horizon: int,
    direction: int,
    limit_threshold: float,
    oos_start: pd.Timestamp,
    cost_bps: float = 0.0,
    slippage_bps: float = 0.0,
    trade_sides: int = 2,
) -> np.ndarray:
    """Build a random-date return pool restricted to one regime."""

    values: list[float] = []
    for frame in price_map.values():
        if "Close" not in frame:
            continue
        close = pd.to_numeric(frame["Close"], errors="coerce").dropna()
        close.index = pd.to_datetime(close.index)
        close = close.sort_index()
        limit_hit = close.pct_change().abs() >= limit_threshold
        idxs = np.flatnonzero(np.asarray(close.index >= pd.Timestamp(oos_start)))
        idxs = idxs[idxs + int(horizon) < len(close)]
        for idx in idxs:
            dt = close.index[idx]
            if regime_at_date(regime_series, dt) != target_regime:
                continue
            bad_limit = bool(limit_hit.iloc[idx])
            if idx + 1 < len(close):
                bad_limit = bad_limit or bool(limit_hit.iloc[idx + 1])
            if bad_limit:
                continue
            ret = _edge_return(close, int(idx), int(horizon), int(direction))
            if ret is not None:
                values.append(ret)
    return np.array(
        apply_signal_costs(
            values,
            cost_bps=cost_bps,
            slippage_bps=slippage_bps,
            trade_sides=trade_sides,
        ),
        dtype=float,
    )


def summarize_regime_alpha_decay(
    events_by_signal: Mapping[str, Sequence[Event]],
    price_map: Mapping[str, pd.DataFrame],
    regime_series: pd.Series | None,
    *,
    limit_threshold: float,
    n: int,
    seed: int,
    oos_start: pd.Timestamp,
    regimes: Sequence[str] = DEFAULT_REGIMES,
    cost_bps: float = 0.0,
    slippage_bps: float = 0.0,
    trade_sides: int = 2,
) -> dict[str, dict[str, Any]]:
    """Compare Alpha Decay OOS edge with same-regime random dates."""

    rng = np.random.default_rng(seed)
    grouped = split_events_by_regime(events_by_signal, regime_series)
    regime_names = list(dict.fromkeys([*regimes, *grouped.keys()]))
    pool_cache: dict[tuple[str, int, int], np.ndarray] = {}
    summaries: dict[str, dict[str, Any]] = {}

    for regime in regime_names:
        by_signal = grouped.get(regime, {})
        actual: list[float] = []
        signal_rows: list[dict[str, Any]] = []
        drawable: list[tuple[str, int, int, int]] = []

        for signal_name, events in by_signal.items():
            returns = [float(event[4]) for event in events]
            actual.extend(returns)
            if events:
                horizon = int(events[0][2])
                direction = int(events[0][3])
                drawable.append((signal_name, len(events), horizon, direction))
                signal_rows.append(
                    {
                        "signal": signal_name,
                        "events": len(events),
                        "edge": float(np.mean(returns)) if returns else None,
                        "horizon": horizon,
                    }
                )

        random_means: list[float] = []
        if actual and drawable:
            for _ in range(max(0, int(n))):
                sample: list[float] = []
                for _, count, horizon, direction in drawable:
                    key = (regime, int(horizon), int(direction))
                    if key not in pool_cache:
                        pool_cache[key] = random_return_pool_by_regime(
                            price_map,
                            regime_series,
                            target_regime=regime,
                            horizon=horizon,
                            direction=direction,
                            limit_threshold=limit_threshold,
                            oos_start=oos_start,
                            cost_bps=cost_bps,
                            slippage_bps=slippage_bps,
                            trade_sides=trade_sides,
                        )
                    pool = pool_cache[key]
                    if len(pool) > 0:
                        sample.extend(rng.choice(pool, size=count, replace=True).tolist())
                if sample:
                    random_means.append(float(np.mean(sample)))

        actual_mean = float(np.mean(actual)) if actual else None
        random_mean = float(np.mean(random_means)) if random_means else None
        p_value = None
        percentile = None
        if actual_mean is not None and random_means:
            random_array = np.array(random_means, dtype=float)
            p_value = float(np.mean(random_array >= actual_mean))
            percentile = float(np.mean(random_array <= actual_mean) * 100.0)

        summaries[regime] = {
            "n_events": len(actual),
            "actual_mean": actual_mean,
            "random_mean": random_mean,
            "p_value": p_value,
            "percentile": percentile,
            "signal_rows": signal_rows,
        }
    return summaries


def regime_summary_rows(summary: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Flatten regime summaries for CSV/report output."""

    rows = []
    for regime, values in summary.items():
        actual = values.get("actual_mean")
        random = values.get("random_mean")
        rows.append(
            {
                "regime": regime,
                "events": int(values.get("n_events", 0) or 0),
                "edge_pct": None if actual is None else float(actual) * 100.0,
                "random_pct": None if random is None else float(random) * 100.0,
                "excess_pct": None if actual is None or random is None else (float(actual) - float(random)) * 100.0,
                "percentile": values.get("percentile"),
                "p_value": values.get("p_value"),
            }
        )
    return rows