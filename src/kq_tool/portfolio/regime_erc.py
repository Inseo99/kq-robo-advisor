"""Regime-conditional ERC allocation with sample-size shrinkage.

This is the v1 implementation for the four-row regime base allocation table.
It deliberately avoids performance optimization. The steps are:

1. Align real monthly asset-class returns with PiT regime labels.
2. Estimate each regime covariance.
3. Shrink regime covariance toward the full-sample covariance with
   kappa = n_regime / (n_regime + n0).
4. Solve bounded ERC inside fixed asset bounds.

The same "small sample retreats to the prior/global estimate" principle is
used in the transition matrix and shadow policy bandit layers.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from kq_tool.regime.regime_labels import REGIMES


DEFAULT_ASSET_BOUNDS: dict[str, tuple[float, float]] = {
    "stocks": (0.05, 0.70),
    "bonds": (0.05, 0.70),
    "gold": (0.03, 0.45),
    "cash": (0.05, 0.60),
}

DEFAULT_REGIME_ASSET_BOUNDS: dict[str, dict[str, tuple[float, float]]] = {
    # Growth-friendly: equity-led, but no single-asset concentration.
    "골디락스": {
        "stocks": (0.35, 0.70),
        "bonds": (0.10, 0.35),
        "gold": (0.03, 0.15),
        "cash": (0.05, 0.25),
    },
    # Inflationary growth: equities plus inflation hedges; bonds capped.
    "리플레이션": {
        "stocks": (0.30, 0.60),
        "bonds": (0.05, 0.25),
        "gold": (0.10, 0.35),
        "cash": (0.05, 0.30),
    },
    # Inflationary slowdown: defensive cash/gold first; bonds not allowed to dominate.
    "스태그플레이션": {
        "stocks": (0.05, 0.25),
        "bonds": (0.05, 0.30),
        "gold": (0.20, 0.45),
        "cash": (0.20, 0.50),
    },
    # Deflationary slowdown: duration and cash protection.
    "디플레이션": {
        "stocks": (0.05, 0.25),
        "bonds": (0.35, 0.70),
        "gold": (0.05, 0.20),
        "cash": (0.15, 0.45),
    },
}


@dataclass(frozen=True)
class RegimeERCConfig:
    n0: float = 36.0
    ridge: float = 1e-8
    bounds: Mapping[str, tuple[float, float]] | None = None
    regime_bounds: Mapping[str, Mapping[str, tuple[float, float]]] | None = None


def _nearest_psd(cov: pd.DataFrame, ridge: float) -> pd.DataFrame:
    arr = cov.to_numpy(dtype=float)
    arr = 0.5 * (arr + arr.T)
    eigvals = np.linalg.eigvalsh(arr)
    min_eig = float(eigvals.min()) if len(eigvals) else 0.0
    if min_eig < ridge:
        arr = arr + np.eye(arr.shape[0]) * (ridge - min_eig)
    return pd.DataFrame(arr, index=cov.index, columns=cov.columns)


def shrink_covariance(
    regime_returns: pd.DataFrame,
    global_returns: pd.DataFrame,
    n0: float = 36.0,
    ridge: float = 1e-8,
) -> tuple[pd.DataFrame, float]:
    """Return kappa-shrunk covariance and its sample weight."""

    n = int(len(regime_returns.dropna(how="all")))
    assets = list(global_returns.columns)
    global_cov = global_returns[assets].cov()
    if n >= 2:
        regime_cov = regime_returns[assets].cov()
    else:
        regime_cov = global_cov.copy()
    regime_cov = regime_cov.reindex(index=assets, columns=assets).fillna(global_cov)
    kappa = n / (n + float(n0)) if n0 > 0 else 1.0
    cov = kappa * regime_cov + (1.0 - kappa) * global_cov
    return _nearest_psd(cov, ridge), float(kappa)


def _initial_weights(assets: list[str], bounds: Mapping[str, tuple[float, float]]) -> np.ndarray:
    lows = np.array([bounds[a][0] for a in assets], dtype=float)
    highs = np.array([bounds[a][1] for a in assets], dtype=float)
    residual = 1.0 - lows.sum()
    capacity = highs - lows
    if residual < -1e-12 or capacity.sum() < residual - 1e-12:
        raise ValueError("Asset bounds cannot sum to 1")
    if residual <= 0:
        return lows / lows.sum()
    return lows + residual * capacity / capacity.sum()


def risk_contributions(weights: np.ndarray, cov: np.ndarray) -> tuple[np.ndarray, float]:
    sigma = float(np.sqrt(max(weights @ cov @ weights, 0.0)))
    if sigma <= 1e-12:
        return np.zeros_like(weights), 0.0
    marginal = cov @ weights
    return weights * marginal / sigma, sigma


def bounded_erc_weights(
    cov: pd.DataFrame,
    bounds: Mapping[str, tuple[float, float]] | None = None,
) -> pd.Series:
    """Solve bounded equal-risk-contribution weights."""

    assets = list(cov.columns)
    b = dict(bounds or DEFAULT_ASSET_BOUNDS)
    missing = [asset for asset in assets if asset not in b]
    if missing:
        raise ValueError(f"Missing bounds for assets: {missing}")
    x0 = _initial_weights(assets, b)
    arr = cov.to_numpy(dtype=float)
    opt_bounds = [b[a] for a in assets]
    constraints = [{"type": "eq", "fun": lambda w: float(w.sum() - 1.0)}]

    def objective(w: np.ndarray) -> float:
        rc, sigma = risk_contributions(w, arr)
        target = sigma / len(w)
        return float(np.sum((rc - target) ** 2))

    result = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=opt_bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    weights = result.x if result.success else x0
    weights = np.clip(weights, [b[a][0] for a in assets], [b[a][1] for a in assets])
    weights = weights / weights.sum()
    return pd.Series(weights, index=assets, name="weight")


def build_regime_erc_allocations(
    returns: pd.DataFrame,
    labels: pd.Series,
    config: RegimeERCConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return regime x asset weights and an audit table."""

    cfg = config or RegimeERCConfig()
    base_bounds = dict(cfg.bounds or DEFAULT_ASSET_BOUNDS)
    regime_bounds = cfg.regime_bounds or DEFAULT_REGIME_ASSET_BOUNDS
    assets = list(base_bounds)
    rets = returns[assets].dropna(how="all").copy()
    rets.index = pd.to_datetime(rets.index).to_period("M").to_timestamp("M")
    labs = labels.dropna().astype(str).copy()
    labs.index = pd.to_datetime(labs.index).to_period("M").to_timestamp("M")
    common = rets.index.intersection(labs.index)
    if len(common) < 24:
        raise ValueError("Not enough common return/regime months")
    rets = rets.loc[common]
    labs = labs.loc[common]

    weight_rows: dict[str, pd.Series] = {}
    audit_rows: list[dict[str, object]] = []
    for regime in REGIMES:
        bounds = dict(regime_bounds.get(regime, base_bounds))
        subset = rets[labs == regime]
        cov, kappa = shrink_covariance(subset, rets, cfg.n0, cfg.ridge)
        weights = bounded_erc_weights(cov, bounds)
        rc, sigma = risk_contributions(weights.to_numpy(dtype=float), cov.to_numpy(dtype=float))
        rc_share = rc / rc.sum() if rc.sum() > 0 else np.zeros_like(rc)
        weight_rows[regime] = weights
        audit_rows.append(
            {
                "regime": regime,
                "n_months": int(len(subset)),
                "kappa": kappa,
                "portfolio_vol_monthly": sigma,
                "max_weight": float(weights.max()),
                "min_weight": float(weights.min()),
                "max_rc_share": float(rc_share.max()) if len(rc_share) else 0.0,
                "min_rc_share": float(rc_share.min()) if len(rc_share) else 0.0,
                "bounds": bounds,
                "config": asdict(cfg),
            }
        )
    weights_df = pd.DataFrame(weight_rows).T[assets]
    audit_df = pd.DataFrame(audit_rows)
    return weights_df, audit_df
