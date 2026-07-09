"""Point-in-Time transition matrix utilities for TEA validation.

The production UI may use the full historical transition matrix for today's
recommendation, because all past labels are known today. Backtests are
different: at month ``t`` they must estimate the transition matrix only from
labels observable up to ``t``. This module provides that expanding-window
estimator and the small audit helpers used by validation gates.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .regime_labels import REGIMES

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_LABEL_PATH = ROOT / "data" / "regime" / "labels.csv"
LEGACY_LABEL_PATH = ROOT / "data" / "macro" / "regime_labels.csv"


@dataclass(frozen=True)
class TransitionPitConfig:
    """Pre-registered transition-estimation settings."""

    burn_in_months: int = 60
    n0: float = 12.0
    prior_sticky: float = 0.85
    horizon_months: int = 3


def _month_end_index(index: Iterable[object]) -> pd.DatetimeIndex:
    dates = pd.to_datetime(pd.Index(index))
    return pd.DatetimeIndex(dates).to_period("M").to_timestamp("M")


def load_regime_history(path: str | Path | None = None) -> pd.Series:
    """Load the canonical real-data monthly regime labels.

    Expected schema: ``date,regime``. The function intentionally validates
    aggressively so a stale or malformed label file cannot silently pass a
    transition gate.
    """

    target = Path(path) if path is not None else DEFAULT_LABEL_PATH
    if not target.exists() and path is None and LEGACY_LABEL_PATH.exists():
        target = LEGACY_LABEL_PATH
    if not target.exists():
        raise FileNotFoundError(
            f"Regime label file not found: {target}. "
            "Prepare data/regime/labels.csv with columns date,regime."
        )

    df = pd.read_csv(target)
    if not {"date", "regime"} <= set(df.columns):
        raise ValueError(f"Regime label file must have date,regime columns: {target}")
    labels = pd.Series(
        df["regime"].astype(str).to_numpy(),
        index=_month_end_index(df["date"]),
        name="regime",
    ).sort_index()
    labels = labels[~labels.index.duplicated(keep="last")]
    unknown = sorted(set(labels.dropna()) - set(REGIMES))
    if unknown:
        raise ValueError(f"Unknown regime labels in {target}: {unknown}")
    if labels.empty:
        raise ValueError(f"Regime label file is empty: {target}")

    expected = pd.date_range(labels.index[0], labels.index[-1], freq="ME")
    missing = expected.difference(labels.index)
    if len(missing):
        preview = ", ".join(d.strftime("%Y-%m") for d in missing[:5])
        raise ValueError(f"Missing monthly regime labels: {preview}")
    return labels.reindex(expected)


def sticky_prior_matrix(
    regimes: list[str] | tuple[str, ...] = REGIMES,
    sticky: float = 0.85,
) -> pd.DataFrame:
    """Return a sticky Markov prior used before enough transitions exist."""

    names = list(regimes)
    n = len(names)
    if n < 2:
        raise ValueError("Need at least two regimes")
    sticky = float(sticky)
    if not 0.0 < sticky < 1.0:
        raise ValueError("prior_sticky must be between 0 and 1")
    off_diag = (1.0 - sticky) / (n - 1)
    arr = np.full((n, n), off_diag, dtype=float)
    np.fill_diagonal(arr, sticky)
    return pd.DataFrame(arr, index=names, columns=names)


def transition_counts(labels: pd.Series, regimes: list[str] | tuple[str, ...] = REGIMES) -> pd.DataFrame:
    """Count monthly label transitions."""

    names = list(regimes)
    clean = labels.dropna().astype(str)
    counts = pd.DataFrame(0.0, index=names, columns=names)
    for before, after in zip(clean.iloc[:-1], clean.iloc[1:]):
        if before in counts.index and after in counts.columns:
            counts.loc[before, after] += 1.0
    return counts


def posterior_transition_matrix(
    counts: pd.DataFrame,
    prior: pd.DataFrame | None = None,
    n0: float = 12.0,
) -> pd.DataFrame:
    """Dirichlet posterior mean: (n0 * prior + counts) / (n0 + row_count)."""

    if prior is None:
        prior = sticky_prior_matrix(list(counts.index))
    prior = prior.reindex(index=counts.index, columns=counts.columns).astype(float)
    if prior.isna().any().any():
        raise ValueError("Prior matrix does not cover all regimes")
    n0 = float(n0)
    if n0 <= 0:
        raise ValueError("n0 must be positive")
    numer = n0 * prior + counts.astype(float)
    denom = n0 + counts.sum(axis=1)
    probs = numer.div(denom, axis=0)
    probs = probs.div(probs.sum(axis=1), axis=0)
    return probs


def full_sample_transition_matrix(
    labels: pd.Series,
    config: TransitionPitConfig | None = None,
) -> pd.DataFrame:
    cfg = config or TransitionPitConfig()
    prior = sticky_prior_matrix(REGIMES, cfg.prior_sticky)
    counts = transition_counts(labels, REGIMES)
    return posterior_transition_matrix(counts, prior, cfg.n0)


def expanding_transition_matrices(
    labels: pd.Series,
    config: TransitionPitConfig | None = None,
) -> dict[pd.Timestamp, pd.DataFrame]:
    """Return P_t estimated from labels available up to each month t."""

    cfg = config or TransitionPitConfig()
    clean = labels.dropna().astype(str).sort_index()
    if len(clean) < cfg.burn_in_months:
        raise ValueError(
            f"Need at least burn_in_months labels: {len(clean)} < {cfg.burn_in_months}"
        )
    prior = sticky_prior_matrix(REGIMES, cfg.prior_sticky)
    matrices: dict[pd.Timestamp, pd.DataFrame] = {}
    for i in range(cfg.burn_in_months - 1, len(clean)):
        history = clean.iloc[: i + 1]
        counts = transition_counts(history, REGIMES)
        matrices[pd.Timestamp(history.index[-1])] = posterior_transition_matrix(
            counts, prior, cfg.n0
        )
    return matrices


def p_power(matrix: pd.DataFrame, horizon_months: int = 3) -> pd.DataFrame:
    if horizon_months < 1:
        raise ValueError("horizon_months must be >= 1")
    arr = np.linalg.matrix_power(matrix.to_numpy(dtype=float), int(horizon_months))
    return pd.DataFrame(arr, index=matrix.index, columns=matrix.columns)


def lookahead_distance_summary(
    labels: pd.Series,
    config: TransitionPitConfig | None = None,
) -> dict[str, object]:
    """Quantify how much full-sample P would differ from PiT P through time."""

    cfg = config or TransitionPitConfig()
    matrices = expanding_transition_matrices(labels, cfg)
    full_p3 = p_power(full_sample_transition_matrix(labels, cfg), cfg.horizon_months)
    rows: list[dict[str, object]] = []
    clean = labels.dropna().astype(str)
    for date, pit_p in matrices.items():
        regime = str(clean.loc[date])
        pit_row = p_power(pit_p, cfg.horizon_months).loc[regime]
        full_row = full_p3.loc[regime]
        rows.append(
            {
                "date": date,
                "regime": regime,
                "l1": float((pit_row - full_row).abs().sum()),
            }
        )
    distances = pd.DataFrame(rows).set_index("date")
    n = len(distances)
    third = max(n // 3, 1)
    early = float(distances["l1"].iloc[:third].mean())
    late = float(distances["l1"].iloc[-third:].mean())
    return {
        "config": asdict(cfg),
        "n_months": int(len(clean)),
        "n_eval_months": int(n),
        "early_l1_mean": early,
        "late_l1_mean": late,
        "max_l1": float(distances["l1"].max()),
        "decayed": bool(late <= early + 1e-12),
        "distances": distances,
    }


def transition_audit_payload(
    labels: pd.Series | None = None,
    *,
    data_source: str = "real",
    config: TransitionPitConfig | None = None,
) -> dict[str, object]:
    """Build a compact JSON-serializable audit record for dashboards."""

    cfg = config or TransitionPitConfig()
    labels = load_regime_history() if labels is None else labels
    matrices = expanding_transition_matrices(labels, cfg)
    full = full_sample_transition_matrix(labels, cfg)
    final_date = max(matrices)
    final_equal = bool(np.allclose(matrices[final_date].values, full.values, atol=1e-12))
    summary = lookahead_distance_summary(labels, cfg)
    return {
        "gate": "transition_pit",
        "data_source": data_source,
        "status": "PASS" if final_equal else "FAIL",
        "config": asdict(cfg),
        "label_start": str(labels.dropna().index[0].date()),
        "label_end": str(labels.dropna().index[-1].date()),
        "n_months": int(labels.dropna().shape[0]),
        "final_expanding_equals_full": final_equal,
        "early_l1_mean": round(float(summary["early_l1_mean"]), 6),
        "late_l1_mean": round(float(summary["late_l1_mean"]), 6),
        "max_l1": round(float(summary["max_l1"]), 6),
        "lookahead_impact_decayed": bool(summary["decayed"]),
    }
