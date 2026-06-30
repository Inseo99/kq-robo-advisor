"""Risk-based portfolio allocation helpers."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


def _normalize_array(weights: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(weights, dtype=float), 0, None)
    total = clipped.sum()
    if total <= 0:
        return np.ones_like(clipped) / len(clipped)
    return clipped / total


def risk_contributions(weights: np.ndarray, cov: np.ndarray) -> tuple[np.ndarray, float]:
    """Return asset risk contributions and portfolio volatility."""

    weights = np.asarray(weights, dtype=float)
    cov = np.asarray(cov, dtype=float)
    sigma_p = float(np.sqrt(weights @ cov @ weights))
    if sigma_p <= 1e-12:
        return np.zeros_like(weights), 0.0
    marginal = cov @ weights
    contributions = weights * marginal / sigma_p
    return contributions, sigma_p


def gmv_weights(cov: np.ndarray, cap: float = 0.5) -> np.ndarray:
    """Global Minimum Variance allocation."""

    cov = np.asarray(cov, dtype=float)
    n = cov.shape[0]
    x0 = np.ones(n) / n
    bounds = [(0.0, cap)] * n
    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    result = minimize(
        lambda w: w @ cov @ w,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-10},
    )
    return _normalize_array(result.x if result.success else x0)


def mdp_weights(cov: np.ndarray, vols: np.ndarray, cap: float = 0.5) -> np.ndarray:
    """Most Diversified Portfolio allocation."""

    cov = np.asarray(cov, dtype=float)
    vols = np.asarray(vols, dtype=float)
    n = cov.shape[0]
    x0 = np.ones(n) / n
    bounds = [(0.0, cap)] * n
    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]

    def negative_diversification_ratio(w: np.ndarray) -> float:
        numerator = w @ vols
        denominator = np.sqrt(max(w @ cov @ w, 1e-12))
        return float(-numerator / denominator)

    result = minimize(
        negative_diversification_ratio,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-10},
    )
    return _normalize_array(result.x if result.success else x0)


def erc_weights(cov: np.ndarray, cap: float = 0.5) -> np.ndarray:
    """Equal Risk Contribution / Risk Parity allocation."""

    cov = np.asarray(cov, dtype=float)
    n = cov.shape[0]
    x0 = np.ones(n) / n
    bounds = [(1e-6, cap)] * n
    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]

    def objective(w: np.ndarray) -> float:
        contributions, sigma_p = risk_contributions(w, cov)
        target = sigma_p / n
        return float(np.sum((contributions - target) ** 2))

    result = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    return _normalize_array(result.x if result.success else x0)


def diversification_ratio(weights: np.ndarray, cov: np.ndarray, vols: np.ndarray) -> float:
    """Return portfolio diversification ratio."""

    weights = np.asarray(weights, dtype=float)
    vols = np.asarray(vols, dtype=float)
    cov = np.asarray(cov, dtype=float)
    numerator = weights @ vols
    denominator = np.sqrt(max(weights @ cov @ weights, 1e-12))
    return float(numerator / denominator)


_gmv_weights = gmv_weights
_mdp_weights = mdp_weights
_erc_weights = erc_weights
_diversification_ratio = diversification_ratio
