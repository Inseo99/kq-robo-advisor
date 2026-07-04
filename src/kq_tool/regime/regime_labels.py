"""Point-in-Time safe regime label generation.

This module builds official 4-quadrant macro regime labels on top of
``macro_data.get_observable_panel()``. It fixes three weaknesses in the legacy
rule labels:

1. Inflation axis: use CPI YoY momentum instead of yield-curve spread.
2. Growth threshold: use an expanding median instead of a full-sample median.
3. Hysteresis: require a new quadrant to persist before confirming a switch.

The output is intended to become the target label for the forecasting model.
For now it is isolated so the legacy app remains runnable while the stricter
label pipeline is validated.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .macro_data import get_observable_panel

__all__ = [
    "LABEL_AXIS_KEYS",
    "MODEL_FEATURE_EXCLUDE_KEYS",
    "REGIMES",
    "REGIME_TO_INT",
    "LabelConfig",
    "apply_confirmation",
    "make_candidate_labels",
    "make_regime_labels",
    "next_quarter_transition_probs",
    "regime_diagnostics",
    "transition_matrix",
]

REGIMES = ["골디락스", "리플레이션", "스태그플레이션", "디플레이션"]
REGIME_TO_INT = {regime: i for i, regime in enumerate(REGIMES)}

LABEL_AXIS_KEYS = ("gdp_qoq", "cpi_yoy")
MODEL_FEATURE_EXCLUDE_KEYS = frozenset(LABEL_AXIS_KEYS)


@dataclass(frozen=True)
class LabelConfig:
    """Configuration for PiT-safe regime label generation.

    Parameters should be fixed using label quality diagnostics, not portfolio
    returns. If several configurations are tried, count those attempts in DSR or
    another multiple-testing correction.
    """

    growth_key: str = "gdp_qoq"
    inflation_key: str = "cpi_yoy"
    warmup_months: int = 24
    momentum_window: int = 6
    confirm_months: int = 3
    laplace_alpha: float = 1.0


def make_candidate_labels(
    panel: pd.DataFrame,
    cfg: LabelConfig = LabelConfig(),
) -> pd.Series:
    """Return raw monthly quadrant labels before hysteresis.

    Growth is high when observable GDP growth is above its expanding median.
    Inflation is rising when CPI YoY has positive ``momentum_window``-month
    momentum. No full-sample statistic is used.
    """

    missing = [key for key in (cfg.growth_key, cfg.inflation_key) if key not in panel]
    if missing:
        raise KeyError(f"Missing label inputs: {missing}")

    growth = pd.to_numeric(panel[cfg.growth_key], errors="coerce")
    inflation = pd.to_numeric(panel[cfg.inflation_key], errors="coerce")

    growth_ref = growth.expanding(min_periods=cfg.warmup_months).median()
    growth_up = growth > growth_ref

    inflation_momentum = inflation.diff(cfg.momentum_window)
    inflation_up = inflation_momentum > 0

    valid = growth.notna() & growth_ref.notna() & inflation_momentum.notna()

    labels = pd.Series(pd.NA, index=panel.index, dtype="object", name="candidate")
    labels[valid & growth_up & ~inflation_up] = "골디락스"
    labels[valid & growth_up & inflation_up] = "리플레이션"
    labels[valid & ~growth_up & inflation_up] = "스태그플레이션"
    labels[valid & ~growth_up & ~inflation_up] = "디플레이션"
    return labels


def apply_confirmation(candidates: pd.Series, confirm_months: int) -> pd.Series:
    """Confirm a regime switch only after consecutive candidate observations.

    The loop runs strictly from past to future, so the smoothing itself cannot
    introduce look-ahead. ``confirm_months=1`` reproduces the raw candidate path.
    """

    if confirm_months < 1:
        raise ValueError("confirm_months must be >= 1")

    official: list[object] = []
    current: object = None
    streak_label: object = None
    streak = 0

    for candidate in candidates:
        if pd.isna(candidate):
            official.append(current if current is not None else pd.NA)
            continue

        if candidate == streak_label:
            streak += 1
        else:
            streak_label = candidate
            streak = 1

        if current is None:
            if streak >= confirm_months:
                current = candidate
        elif candidate != current and streak >= confirm_months:
            current = candidate

        official.append(current if current is not None else pd.NA)

    return pd.Series(official, index=candidates.index, dtype="object", name="regime")


def make_regime_labels(
    cfg: LabelConfig = LabelConfig(),
    panel: pd.DataFrame | None = None,
    asof: pd.Timestamp | str | None = None,
    return_frame: bool = False,
) -> pd.Series | pd.DataFrame:
    """Build official monthly regime labels.

    If ``panel`` is omitted, the required series are loaded through
    ``get_observable_panel()``, which applies publication lags and optional
    ``asof`` truncation. ``return_frame=True`` includes diagnostic columns useful
    for charting and manual review.
    """

    if panel is None:
        panel = get_observable_panel([cfg.growth_key, cfg.inflation_key], asof=asof)
    elif asof is not None:
        cutoff = pd.Timestamp(asof).to_period("M").to_timestamp("M")
        panel = panel.loc[:cutoff]

    candidates = make_candidate_labels(panel, cfg)
    official = apply_confirmation(candidates, cfg.confirm_months)

    if not return_frame:
        return official

    out = panel[[cfg.growth_key, cfg.inflation_key]].copy()
    out["growth_ref"] = pd.to_numeric(panel[cfg.growth_key], errors="coerce").expanding(
        min_periods=cfg.warmup_months
    ).median()
    out["inflation_momentum"] = pd.to_numeric(
        panel[cfg.inflation_key], errors="coerce"
    ).diff(cfg.momentum_window)
    out["candidate"] = candidates
    out["regime"] = official
    return out


def _runs(labels: pd.Series) -> list[tuple[str, int]]:
    runs: list[tuple[str, int]] = []
    for label in labels.dropna():
        if runs and runs[-1][0] == label:
            runs[-1] = (runs[-1][0], runs[-1][1] + 1)
        else:
            runs.append((str(label), 1))
    return runs


def regime_diagnostics(labels: pd.Series) -> dict:
    """Return regime duration and share diagnostics for review."""

    runs = _runs(labels)
    durations = pd.Series([duration for _, duration in runs], dtype=float)
    clean = labels.dropna()

    by_regime = {}
    for regime in REGIMES:
        regime_durations = [duration for label, duration in runs if label == regime]
        if regime_durations:
            by_regime[regime] = float(np.mean(regime_durations) / 3.0)

    return {
        "n_months": int(len(clean)),
        "n_transitions": max(len(runs) - 1, 0),
        "avg_duration_quarters": float(durations.mean() / 3.0) if len(runs) else np.nan,
        "avg_duration_by_regime_q": by_regime,
        "regime_share": clean.value_counts(normalize=True).round(3).to_dict(),
    }


def transition_matrix(
    labels: pd.Series,
    alpha: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return monthly transition probabilities and raw transition counts."""

    clean = labels.dropna()
    counts = pd.DataFrame(0, index=REGIMES, columns=REGIMES, dtype=int)
    for before, after in zip(clean.iloc[:-1], clean.iloc[1:]):
        if before in counts.index and after in counts.columns:
            counts.loc[before, after] += 1

    smoothed = counts.astype(float) + float(alpha)
    probs = smoothed.div(smoothed.sum(axis=1), axis=0)
    return probs, counts


def next_quarter_transition_probs(
    labels: pd.Series,
    alpha: float = 1.0,
) -> pd.DataFrame:
    """Return next-quarter transition probabilities as monthly matrix P^3."""

    probs, _ = transition_matrix(labels, alpha=alpha)
    p3 = np.linalg.matrix_power(probs.to_numpy(dtype=float), 3)
    return pd.DataFrame(p3, index=REGIMES, columns=REGIMES)

