"""Residual momentum research helper.

Blitz-Huij-Martens style residual momentum:
1. Regress each stock's excess return on FF factors using only data visible at
   the formation month.
2. Rank stocks by the recent residual momentum, skipping the latest month.
3. Apply the selected portfolio to the next month's return.

The module is pure: callers provide monthly stock returns and factor returns.
It does not read price files directly.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

FACTOR_COLS: tuple[str, ...] = ("MKT", "SMB", "HML")


@dataclass(frozen=True)
class ResMomConfig:
    est_window: int = 36
    mom_window: int = 12
    skip: int = 1
    top_n: int = 20
    min_obs: int = 30
    standardize: bool = True


@dataclass
class ResMomBacktestResult:
    portfolio_returns: pd.Series
    holdings: pd.DataFrame
    signal_corr_with_plain: pd.Series
    config: dict[str, Any]
    fp_json: str

    def to_payload(self) -> dict[str, Any]:
        mean_corr = self.signal_corr_with_plain.mean()
        return {
            "portfolio_returns": {
                str(pd.Timestamp(k).date()): round(float(v), 8)
                for k, v in self.portfolio_returns.items()
            },
            "n_rebalances": int(len(self.holdings)),
            "signal_corr_with_plain_mean": (
                None if pd.isna(mean_corr) else round(float(mean_corr), 4)
            ),
            "config": self.config,
            "fp_json": self.fp_json,
        }


def _fingerprint(obj: Any) -> str:
    def norm(value: Any) -> Any:
        if isinstance(value, pd.DataFrame):
            clean = value.copy()
            return {
                "type": "df",
                "shape": list(clean.shape),
                "index": [str(clean.index[0]), str(clean.index[-1])] if len(clean) else [],
                "columns": list(map(str, clean.columns)),
                "sha": hashlib.sha256(
                    pd.util.hash_pandas_object(clean, index=True).values.tobytes()
                ).hexdigest(),
            }
        if isinstance(value, pd.Series):
            return norm(value.to_frame())
        if isinstance(value, dict):
            return {str(k): norm(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
        if isinstance(value, (list, tuple)):
            return [norm(v) for v in value]
        return value

    payload = json.dumps(norm(obj), ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _factor_columns(factors: pd.DataFrame) -> list[str]:
    if all(col in factors.columns for col in FACTOR_COLS):
        return list(FACTOR_COLS)
    legacy = ["MKT_RF", "SMB", "HML"]
    if all(col in factors.columns for col in legacy):
        return legacy
    raise ValueError("FF3 factor columns not found. Expected MKT/SMB/HML or MKT_RF/SMB/HML.")


def _risk_free(factors: pd.DataFrame) -> pd.Series:
    if "RF" in factors.columns:
        return pd.to_numeric(factors["RF"], errors="coerce").fillna(0.0)
    return pd.Series(0.0, index=factors.index)


def align_monthly_inputs(
    returns: pd.DataFrame,
    factors: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    r = returns.copy()
    f = factors.copy()
    r.index = pd.to_datetime(r.index).to_period("M").to_timestamp("M")
    f.index = pd.to_datetime(f.index).to_period("M").to_timestamp("M")
    r = r.sort_index().apply(pd.to_numeric, errors="coerce")
    f = f.sort_index().apply(pd.to_numeric, errors="coerce")
    common = r.index.intersection(f.index)
    return r.loc[common], f.loc[common]


def compute_residual_momentum_signal(
    returns: pd.DataFrame,
    factors: pd.DataFrame,
    asof: pd.Timestamp,
    config: ResMomConfig | None = None,
) -> pd.Series:
    cfg = config or ResMomConfig()
    returns, factors = align_monthly_inputs(returns, factors)
    asof = pd.Timestamp(asof).to_period("M").to_timestamp("M")

    window = returns.loc[:asof].tail(cfg.est_window)
    if len(window) < cfg.min_obs:
        return pd.Series(dtype=float)

    factor_window = factors.reindex(window.index)
    factor_cols = _factor_columns(factor_window)
    if factor_window[factor_cols].isna().any().any():
        raise ValueError(f"{asof.date()} 회귀창에 FF 팩터 결측이 있습니다.")

    x = np.column_stack([np.ones(len(window)), factor_window[factor_cols].to_numpy(dtype=float)])
    rf = _risk_free(factor_window).to_numpy(dtype=float).reshape(-1, 1)

    total_obs = len(window)
    if cfg.mom_window + cfg.skip > total_obs:
        return pd.Series(dtype=float)
    mom_mask = np.zeros(total_obs, dtype=bool)
    mom_mask[total_obs - cfg.mom_window : total_obs - cfg.skip] = True

    signals: dict[str, float] = {}
    for code in window.columns:
        y = window[code].to_numpy(dtype=float).reshape(-1, 1) - rf
        valid = np.isfinite(y).ravel()
        if not valid[-1] or valid.sum() < cfg.min_obs:
            continue
        if not valid[mom_mask].all():
            continue

        xv = x[valid]
        yv = y[valid]
        beta, *_ = np.linalg.lstsq(xv, yv, rcond=None)
        residuals = np.full(total_obs, np.nan)
        residuals[valid] = (yv - xv @ beta).ravel()
        segment = residuals[mom_mask]
        if cfg.standardize:
            sd = np.nanstd(segment, ddof=1)
            if not np.isfinite(sd) or sd < 1e-12:
                continue
            signals[str(code)] = float(np.nanmean(segment) / sd)
        else:
            signals[str(code)] = float(np.nansum(segment))

    return pd.Series(signals, name=f"resmom_{asof.date()}").sort_values(ascending=False)


def compute_plain_momentum_signal(
    returns: pd.DataFrame,
    asof: pd.Timestamp,
    config: ResMomConfig | None = None,
) -> pd.Series:
    cfg = config or ResMomConfig()
    returns = returns.copy()
    returns.index = pd.to_datetime(returns.index).to_period("M").to_timestamp("M")
    asof = pd.Timestamp(asof).to_period("M").to_timestamp("M")
    window = returns.loc[:asof].tail(cfg.mom_window)
    if len(window) < cfg.mom_window:
        return pd.Series(dtype=float)
    obs = window.iloc[: len(window) - cfg.skip]
    ok = obs.notna().all() & window.iloc[-1].notna()
    return ((1.0 + obs).prod() - 1.0).where(ok).dropna().sort_values(ascending=False)


def run_backtest(
    returns: pd.DataFrame,
    factors: pd.DataFrame,
    config: ResMomConfig | None = None,
    start: str | None = None,
    end: str | None = None,
) -> ResMomBacktestResult:
    cfg = config or ResMomConfig()
    returns, factors = align_monthly_inputs(returns, factors)
    dates = returns.index
    if start:
        dates = dates[dates >= pd.Timestamp(start)]
    if end:
        dates = dates[dates <= pd.Timestamp(end)]

    formation_dates = [
        date for date in dates if returns.index.get_loc(date) >= cfg.est_window - 1
    ]
    portfolio_returns: dict[pd.Timestamp, float] = {}
    holdings: dict[pd.Timestamp, pd.Series] = {}
    correlations: dict[pd.Timestamp, float] = {}

    for idx, date in enumerate(formation_dates[:-1]):
        signal = compute_residual_momentum_signal(returns, factors, date, cfg)
        if len(signal) < cfg.top_n:
            continue
        picks = signal.nlargest(cfg.top_n).index
        weights = pd.Series(1.0 / len(picks), index=picks)
        holdings[date] = weights

        plain = compute_plain_momentum_signal(returns, date, cfg)
        common = signal.index.intersection(plain.index)
        if len(common) >= 10:
            correlations[date] = float(signal.loc[common].corr(plain.loc[common], method="spearman"))

        next_date = formation_dates[idx + 1]
        next_returns = returns.loc[next_date, picks]
        portfolio_returns[next_date] = float(next_returns.fillna(0.0).mean())

    return ResMomBacktestResult(
        portfolio_returns=pd.Series(portfolio_returns).sort_index(),
        holdings=pd.DataFrame(holdings).T.fillna(0.0),
        signal_corr_with_plain=pd.Series(correlations).sort_index(),
        config=asdict(cfg),
        fp_json=_fingerprint(
            {
                "returns": returns,
                "factors": factors,
                "config": asdict(cfg),
                "start": start or "",
                "end": end or "",
            }
        ),
    )

