"""Fama-French style factor-regression helpers.

The project uses this module to define alpha as the part of portfolio return
that is not explained by standard market/style factors.

Expected factor columns:
    MKT: market excess return
    SMB: small minus big
    HML: high book-to-market minus low book-to-market
    RMW: robust profitability minus weak profitability
    CMA: conservative investment minus aggressive investment
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import sqrt

import numpy as np
import pandas as pd


FF3_FACTORS: tuple[str, ...] = ("MKT", "SMB", "HML")
FF5_FACTORS: tuple[str, ...] = ("MKT", "SMB", "HML", "RMW", "CMA")


def _to_numeric_series(values: pd.Series | Sequence[float], name: str) -> pd.Series:
    series = values if isinstance(values, pd.Series) else pd.Series(values, dtype=float)
    series = pd.to_numeric(series, errors="coerce").dropna()
    series.name = series.name or name
    return series.astype(float)


def _risk_free_series(
    index: pd.Index,
    risk_free_rate: float | pd.Series | Sequence[float],
    *,
    freq_per_year: int,
) -> pd.Series:
    if isinstance(risk_free_rate, pd.Series):
        rf = pd.to_numeric(risk_free_rate, errors="coerce")
        return rf.reindex(index).astype(float)
    if isinstance(risk_free_rate, Sequence) and not isinstance(risk_free_rate, (str, bytes)):
        rf = pd.Series(risk_free_rate, index=index[: len(risk_free_rate)], dtype=float)
        return rf.reindex(index).astype(float)
    return pd.Series(float(risk_free_rate) / freq_per_year, index=index, dtype=float)


def align_factor_regression_data(
    portfolio_returns: pd.Series | Sequence[float],
    factor_returns: pd.DataFrame,
    *,
    risk_free_rate: float | pd.Series | Sequence[float] = 0.0,
    freq_per_year: int = 12,
    mkt_is_excess: bool = True,
) -> pd.DataFrame:
    """Return a clean frame with portfolio excess returns and aligned factors."""

    portfolio = _to_numeric_series(portfolio_returns, "portfolio")
    factors = factor_returns.apply(pd.to_numeric, errors="coerce")
    frame = pd.concat([portfolio.rename("portfolio"), factors], axis=1).dropna(how="any")
    if frame.empty:
        return frame

    rf = _risk_free_series(frame.index, risk_free_rate, freq_per_year=freq_per_year)
    frame = frame.join(rf.rename("RF"), how="inner").dropna(how="any")
    frame["portfolio_excess"] = frame["portfolio"] - frame["RF"]
    if "MKT" in frame.columns and not mkt_is_excess:
        frame["MKT"] = frame["MKT"] - frame["RF"]
    return frame


def fama_french_regression(
    portfolio_returns: pd.Series | Sequence[float],
    factor_returns: pd.DataFrame,
    *,
    model: str = "ff5",
    factors: Sequence[str] | None = None,
    risk_free_rate: float | pd.Series | Sequence[float] = 0.0,
    freq_per_year: int = 12,
    mkt_is_excess: bool = True,
) -> dict[str, object]:
    """Run OLS and return alpha, t-statistics, factor loadings, and fit quality."""

    if factors is None:
        factors = FF3_FACTORS if model.lower() in {"ff3", "3", "three"} else FF5_FACTORS

    frame = align_factor_regression_data(
        portfolio_returns,
        factor_returns,
        risk_free_rate=risk_free_rate,
        freq_per_year=freq_per_year,
        mkt_is_excess=mkt_is_excess,
    )
    used_factors = [factor for factor in factors if factor in frame.columns]
    if len(frame) <= len(used_factors) + 1 or not used_factors:
        return {
            "ok": False,
            "reason": "not_enough_observations",
            "n": int(len(frame)),
            "factors": used_factors,
        }

    y = frame["portfolio_excess"].to_numpy(dtype=float)
    x = frame[used_factors].to_numpy(dtype=float)
    design = np.column_stack([np.ones(len(frame)), x])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    fitted = design @ coef
    residuals = y - fitted

    n_obs = len(y)
    n_params = design.shape[1]
    dof = max(n_obs - n_params, 1)
    sigma2 = float((residuals @ residuals) / dof)
    xtx_inv = np.linalg.pinv(design.T @ design)
    se = np.sqrt(np.diag(xtx_inv) * sigma2)
    t_stats = np.divide(coef, se, out=np.zeros_like(coef), where=se > 0)

    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    adj_r2 = 1.0 - (1.0 - r2) * (n_obs - 1) / dof if n_obs > 1 else r2
    alpha_period = float(coef[0])

    return {
        "ok": True,
        "model": "FF3" if tuple(used_factors) == FF3_FACTORS else "FF5",
        "n": int(n_obs),
        "factors": used_factors,
        "alpha_period": alpha_period,
        "alpha_period_pct": alpha_period * 100,
        "alpha_annual": alpha_period * freq_per_year,
        "alpha_annual_pct": alpha_period * freq_per_year * 100,
        "alpha_t": float(t_stats[0]),
        "betas": {factor: float(value) for factor, value in zip(used_factors, coef[1:])},
        "t_stats": {
            "alpha": float(t_stats[0]),
            **{factor: float(value) for factor, value in zip(used_factors, t_stats[1:])},
        },
        "r2": float(r2),
        "adj_r2": float(adj_r2),
        "residual_vol_annual": float(np.std(residuals, ddof=1) * sqrt(freq_per_year)),
    }


def run_factor_regressions(
    portfolios: Mapping[str, pd.Series | Sequence[float]] | pd.DataFrame,
    factor_returns: pd.DataFrame,
    *,
    model: str = "ff5",
    risk_free_rate: float | pd.Series | Sequence[float] = 0.0,
    freq_per_year: int = 12,
    mkt_is_excess: bool = True,
) -> dict[str, dict[str, object]]:
    """Run the same Fama-French regression for multiple portfolio return series."""

    if isinstance(portfolios, pd.DataFrame):
        items = {col: portfolios[col] for col in portfolios.columns}
    else:
        items = dict(portfolios)
    return {
        name: fama_french_regression(
            returns,
            factor_returns,
            model=model,
            risk_free_rate=risk_free_rate,
            freq_per_year=freq_per_year,
            mkt_is_excess=mkt_is_excess,
        )
        for name, returns in items.items()
    }


def cross_sectional_factor_returns(
    returns: pd.DataFrame,
    *,
    market_return: pd.Series | None = None,
    risk_free_rate: float | pd.Series | Sequence[float] = 0.0,
    size: pd.DataFrame | pd.Series | None = None,
    book_to_market: pd.DataFrame | pd.Series | None = None,
    profitability: pd.DataFrame | pd.Series | None = None,
    investment: pd.DataFrame | pd.Series | None = None,
    freq_per_year: int = 12,
    quantile: float = 0.3,
) -> pd.DataFrame:
    """Build simple Korean-market FF-style factors from cross-sectional inputs.

    ``size`` uses small-minus-big, ``book_to_market`` uses high-minus-low,
    ``profitability`` uses robust-minus-weak, and ``investment`` uses
    conservative-minus-aggressive.
    """

    rets = returns.apply(pd.to_numeric, errors="coerce").dropna(how="all")
    if rets.empty:
        return pd.DataFrame(index=rets.index)

    rf = _risk_free_series(rets.index, risk_free_rate, freq_per_year=freq_per_year)
    out = pd.DataFrame(index=rets.index)
    if market_return is not None:
        out["MKT"] = pd.to_numeric(market_return, errors="coerce").reindex(rets.index) - rf
    else:
        out["MKT"] = rets.mean(axis=1) - rf

    def characteristic_frame(values: pd.DataFrame | pd.Series | None) -> pd.DataFrame | None:
        if values is None:
            return None
        if isinstance(values, pd.Series):
            return pd.DataFrame([values] * len(rets), index=rets.index)
        return values.apply(pd.to_numeric, errors="coerce").reindex(rets.index).ffill()

    def spread(
        values: pd.DataFrame | pd.Series | None,
        *,
        high_minus_low: bool,
    ) -> pd.Series | None:
        chars = characteristic_frame(values)
        if chars is None:
            return None

        rows: list[float] = []
        for dt, row in chars.iterrows():
            available = row.dropna().index.intersection(rets.columns)
            if len(available) < 4 or dt not in rets.index:
                rows.append(np.nan)
                continue
            ranked = row.loc[available].sort_values()
            bucket = max(int(np.floor(len(ranked) * quantile)), 1)
            low_names = ranked.index[:bucket]
            high_names = ranked.index[-bucket:]
            period_returns = rets.loc[dt]
            value = period_returns.loc[high_names].mean() - period_returns.loc[low_names].mean()
            rows.append(float(value if high_minus_low else -value))
        return pd.Series(rows, index=rets.index)

    smb = spread(size, high_minus_low=False)
    if smb is not None:
        out["SMB"] = smb
    hml = spread(book_to_market, high_minus_low=True)
    if hml is not None:
        out["HML"] = hml
    rmw = spread(profitability, high_minus_low=True)
    if rmw is not None:
        out["RMW"] = rmw
    cma = spread(investment, high_minus_low=False)
    if cma is not None:
        out["CMA"] = cma

    return out.dropna(how="all")


def factor_regression_rows(results: Mapping[str, Mapping[str, object]]) -> list[dict[str, object]]:
    """Flatten regression results for CSV/report tables."""

    rows: list[dict[str, object]] = []
    for name, result in results.items():
        row = {
            "portfolio": name,
            "ok": bool(result.get("ok")),
            "model": result.get("model"),
            "n": result.get("n"),
            "alpha_annual_pct": result.get("alpha_annual_pct"),
            "alpha_t": result.get("alpha_t"),
            "r2": result.get("r2"),
        }
        for factor, beta in dict(result.get("betas") or {}).items():
            row[f"beta_{factor}"] = beta
            row[f"t_{factor}"] = dict(result.get("t_stats") or {}).get(factor)
        rows.append(row)
    return rows


_fama_french_regression = fama_french_regression
_run_factor_regressions = run_factor_regressions
_cross_sectional_factor_returns = cross_sectional_factor_returns
