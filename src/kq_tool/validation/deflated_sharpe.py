"""Deflated Sharpe Ratio tools.

The Deflated Sharpe Ratio (DSR) corrects the usual Sharpe significance test for
selection bias, non-normal returns, and multiple trials. All formula inputs are
computed at the observed return frequency; annualized Sharpe values are
reported separately for readability.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy import stats

EULER_MASCHERONI = 0.5772156649015329
PERIODS_PER_YEAR = 12


def _clean(returns: Sequence[float] | pd.Series) -> np.ndarray:
    arr = np.asarray(pd.Series(returns).dropna(), dtype=float)
    if arr.size < 12:
        raise ValueError(f"Too few observations for Sharpe validation (T={arr.size}); need at least 12.")
    return arr


def sharpe_ratio(
    returns: Sequence[float] | pd.Series,
    rf: float | Sequence[float] = 0.0,
    ddof: int = 1,
) -> float:
    """Observed-frequency Sharpe ratio."""
    r = _clean(returns)
    rf_arr = np.broadcast_to(np.asarray(rf, dtype=float), r.shape)
    excess = r - rf_arr
    sd = excess.std(ddof=ddof)
    if sd == 0 or not np.isfinite(sd):
        return 0.0
    return float(excess.mean() / sd)


def annualized_sharpe(
    returns: Sequence[float] | pd.Series,
    rf: float | Sequence[float] = 0.0,
    periods_per_year: int = PERIODS_PER_YEAR,
) -> float:
    """Annualized Sharpe for reporting."""
    return sharpe_ratio(returns, rf) * np.sqrt(periods_per_year)


def probabilistic_sharpe_ratio(
    returns: Sequence[float] | pd.Series,
    sr_benchmark: float = 0.0,
    rf: float | Sequence[float] = 0.0,
) -> float:
    """Probabilistic Sharpe Ratio.

    ``sr_benchmark`` must use the observed return frequency, not annualized SR.
    """
    r = _clean(returns)
    sr_hat = sharpe_ratio(r, rf)
    t = r.size
    g3 = float(stats.skew(r, bias=False))
    g4 = float(stats.kurtosis(r, fisher=False, bias=False))

    denom_sq = 1.0 - g3 * sr_hat + (g4 - 1.0) / 4.0 * sr_hat**2
    if denom_sq <= 0 or not np.isfinite(denom_sq):
        return 0.0
    z = (sr_hat - sr_benchmark) * np.sqrt(t - 1.0) / np.sqrt(denom_sq)
    return float(stats.norm.cdf(z))


def expected_max_sharpe(
    n_trials: int,
    var_trial_sr: float,
    mean_trial_sr: float = 0.0,
) -> float:
    """Expected maximum Sharpe among ``n_trials`` random trials."""
    if n_trials < 1:
        raise ValueError("n_trials must be >= 1.")
    if n_trials == 1 or var_trial_sr <= 0:
        return float(mean_trial_sr)

    gamma = EULER_MASCHERONI
    z1 = stats.norm.ppf(1.0 - 1.0 / n_trials)
    z2 = stats.norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    return float(mean_trial_sr + np.sqrt(var_trial_sr) * ((1.0 - gamma) * z1 + gamma * z2))


def deflated_sharpe_ratio(
    returns: Sequence[float] | pd.Series,
    trial_sharpes: Sequence[float],
    n_trials: int | None = None,
    rf: float | Sequence[float] = 0.0,
) -> dict[str, object]:
    """Compute DSR for one strategy.

    ``trial_sharpes`` should include every candidate strategy Sharpe observed at
    the same frequency as ``returns``. Pass a larger ``n_trials`` when the
    research process tried additional variants that are not present in the
    final table.
    """
    trial_sr = np.asarray(trial_sharpes, dtype=float)
    trial_sr = trial_sr[np.isfinite(trial_sr)]
    if trial_sr.size < 2:
        raise ValueError("trial_sharpes must contain at least two finite values.")

    n = int(n_trials) if n_trials is not None else int(trial_sr.size)
    var_sr = float(trial_sr.var(ddof=1))
    sr0 = expected_max_sharpe(n, var_sr, mean_trial_sr=0.0)
    dsr = probabilistic_sharpe_ratio(returns, sr_benchmark=sr0, rf=rf)

    r = _clean(returns)
    return {
        "sr_monthly": sharpe_ratio(r, rf),
        "sr_annualized": annualized_sharpe(r, rf),
        "skew": float(stats.skew(r, bias=False)),
        "kurtosis_raw": float(stats.kurtosis(r, fisher=False, bias=False)),
        "T": int(r.size),
        "n_trials": n,
        "var_trial_sr": var_sr,
        "expected_max_sr": sr0,
        "expected_max_sr_annualized": sr0 * np.sqrt(PERIODS_PER_YEAR),
        "psr_vs_zero": probabilistic_sharpe_ratio(r, 0.0, rf),
        "dsr": dsr,
        "significant_5pct": bool(dsr >= 0.95),
    }


def deflated_sharpe_table(
    returns_df: pd.DataFrame,
    n_trials: int | None = None,
    rf: float | pd.Series = 0.0,
) -> pd.DataFrame:
    """Build a DSR table for each strategy column."""
    srs = {col: sharpe_ratio(returns_df[col].dropna(), rf) for col in returns_df.columns}
    trial_sr = list(srs.values())
    rows = {
        col: deflated_sharpe_ratio(returns_df[col].dropna(), trial_sr, n_trials=n_trials, rf=rf)
        for col in returns_df.columns
    }
    out = pd.DataFrame(rows).T
    out.index.name = "strategy"
    return out.sort_values("dsr", ascending=False)


def bootstrap_sharpe_ci(
    returns: Sequence[float] | pd.Series,
    n_boot: int = 10_000,
    ci: float = 0.95,
    periods_per_year: int = PERIODS_PER_YEAR,
    block: int = 3,
    seed: int | None = 42,
) -> dict[str, float | int]:
    """Circular block bootstrap CI for annualized Sharpe."""
    r = _clean(returns)
    rng = np.random.default_rng(seed)
    t = r.size
    n_blocks = int(np.ceil(t / block))
    sims = np.empty(n_boot)

    for i in range(n_boot):
        starts = rng.integers(0, t, size=n_blocks)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel() % t
        sample = r[idx[:t]]
        sd = sample.std(ddof=1)
        sims[i] = 0.0 if sd == 0 else sample.mean() / sd * np.sqrt(periods_per_year)

    lo, hi = np.quantile(sims, [(1.0 - ci) / 2.0, 1.0 - (1.0 - ci) / 2.0])
    return {
        "sharpe_annualized": annualized_sharpe(r, periods_per_year=periods_per_year),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "ci_level": float(ci),
        "n_boot": int(n_boot),
        "block": int(block),
    }
