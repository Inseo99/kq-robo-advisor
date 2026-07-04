"""Regime-conditional performance decomposition."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PERIODS_PER_YEAR = 12
MISSING_REGIME_LABELS = {"", "nan", "NaN", "None", "NONE", "<NA>"}


def load_inputs(
    strategy_csv: str | Path,
    regime_csv: str | Path,
    asset_csv: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame | None]:
    """Read monthly strategy returns, regime labels, and optional asset returns."""
    strat = pd.read_csv(strategy_csv, index_col=0, parse_dates=True).sort_index()
    reg = pd.read_csv(regime_csv, index_col=0, parse_dates=True).sort_index()
    regime_col = "regime" if "regime" in reg.columns else reg.columns[0]
    regime = reg[regime_col].astype("string")
    regime = regime.mask(regime.isin(MISSING_REGIME_LABELS)).dropna()

    strat.index = strat.index.to_period("M")
    regime.index = regime.index.to_period("M")

    common = strat.index.intersection(regime.index)
    strat, regime = strat.loc[common], regime.loc[common]

    assets = None
    if asset_csv is not None:
        assets = pd.read_csv(asset_csv, index_col=0, parse_dates=True).sort_index()
        assets.index = assets.index.to_period("M")
        assets = assets.loc[assets.index.intersection(common)]

    return strat, regime.astype(str), assets


def _regime_stats(returns: pd.Series) -> dict[str, float | int]:
    r = returns.dropna()
    n = int(r.size)
    if n == 0:
        return {}

    mu = float(r.mean())
    sd = float(r.std(ddof=1)) if n > 1 else np.nan
    sharpe = np.nan if (not np.isfinite(sd) or sd == 0) else mu / sd * np.sqrt(PERIODS_PER_YEAR)
    t_stat = np.nan if (not np.isfinite(sd) or sd == 0 or n < 2) else mu / (sd / np.sqrt(n))
    cum = (1.0 + r).cumprod()
    max_dd = float((cum / cum.cummax() - 1.0).min()) if n > 1 else 0.0

    return {
        "n_months": n,
        "ann_return": float((1.0 + mu) ** PERIODS_PER_YEAR - 1.0),
        "ann_vol": float(sd * np.sqrt(PERIODS_PER_YEAR)) if np.isfinite(sd) else np.nan,
        "sharpe": float(sharpe) if np.isfinite(sharpe) else np.nan,
        "hit_rate": float((r > 0).mean()),
        "worst_month": float(r.min()),
        "max_dd_concat": max_dd,
        "t_stat_mean": float(t_stat) if np.isfinite(t_stat) else np.nan,
    }


def regime_performance_matrix(
    returns_df: pd.DataFrame,
    regime: pd.Series,
    min_months: int = 12,
) -> pd.DataFrame:
    """Long-format regime x strategy performance table."""
    regime = regime.reindex(returns_df.index).dropna()
    rows = []
    for regime_name, idx in regime.groupby(regime).groups.items():
        for col in returns_df.columns:
            stats_row = _regime_stats(returns_df.loc[idx, col])
            if not stats_row:
                continue
            stats_row.update(
                regime=regime_name,
                strategy=col,
                insufficient=stats_row["n_months"] < min_months,
            )
            rows.append(stats_row)

    cols = [
        "regime",
        "strategy",
        "n_months",
        "insufficient",
        "sharpe",
        "ann_return",
        "ann_vol",
        "hit_rate",
        "worst_month",
        "max_dd_concat",
        "t_stat_mean",
    ]
    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows)[cols].sort_values(["regime", "sharpe"], ascending=[True, False])


def sharpe_pivot(matrix: pd.DataFrame) -> pd.DataFrame:
    """Presentation-friendly regime x strategy annualized Sharpe pivot."""
    if matrix.empty:
        return pd.DataFrame()
    return matrix.pivot(index="regime", columns="strategy", values="sharpe").round(3)


def excess_vs_benchmark(
    returns_df: pd.DataFrame,
    regime: pd.Series,
    benchmark: str,
    min_months: int = 12,
) -> pd.DataFrame:
    """Regime-level excess return and paired t-test versus a benchmark column."""
    if benchmark not in returns_df.columns:
        raise KeyError(f"benchmark '{benchmark}' not found in columns: {list(returns_df.columns)}")

    regime = regime.reindex(returns_df.index).dropna()
    rows = []
    bench = returns_df[benchmark]
    for regime_name, idx in regime.groupby(regime).groups.items():
        b = bench.loc[idx]
        for col in returns_df.columns:
            if col == benchmark:
                continue
            pair = pd.concat([returns_df.loc[idx, col], b], axis=1).dropna()
            n = int(len(pair))
            if n < 2:
                continue
            diff = pair.iloc[:, 0] - pair.iloc[:, 1]
            t_stat, p_value = stats.ttest_rel(pair.iloc[:, 0], pair.iloc[:, 1])
            rows.append(
                {
                    "regime": regime_name,
                    "strategy": col,
                    "benchmark": benchmark,
                    "n_months": n,
                    "insufficient": n < min_months,
                    "ann_excess_pp": float(diff.mean() * PERIODS_PER_YEAR * 100.0),
                    "t_stat": float(t_stat),
                    "p_value": float(p_value),
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=[
                "regime",
                "strategy",
                "benchmark",
                "n_months",
                "insufficient",
                "ann_excess_pp",
                "t_stat",
                "p_value",
            ]
        )
    return (
        pd.DataFrame(rows)
        .sort_values(["regime", "ann_excess_pp"], ascending=[True, False])
        .reset_index(drop=True)
    )


def best_allocation_per_regime(
    matrix: pd.DataFrame,
    metric: str = "sharpe",
) -> pd.DataFrame:
    """Best non-insufficient strategy per regime."""
    if matrix.empty:
        return pd.DataFrame()
    valid = matrix[~matrix["insufficient"]]
    if valid.empty:
        return pd.DataFrame()
    idx = valid.groupby("regime")[metric].idxmax()
    cols = ["regime", "strategy", "n_months", metric, "ann_return", "ann_vol"]
    return valid.loc[idx, cols].reset_index(drop=True)


def run_full_decomposition(
    strategy_csv: str | Path,
    regime_csv: str | Path,
    asset_csv: str | Path | None = None,
    benchmark: str | None = None,
    out_dir: str | Path | None = None,
    min_months: int = 12,
) -> dict[str, pd.DataFrame]:
    """Run the full regime decomposition and optionally save CSV outputs."""
    strat, regime, assets = load_inputs(strategy_csv, regime_csv, asset_csv)
    results: dict[str, pd.DataFrame] = {}

    results["strategy_matrix"] = regime_performance_matrix(strat, regime, min_months)
    results["strategy_sharpe_pivot"] = sharpe_pivot(results["strategy_matrix"])
    results["best_per_regime"] = best_allocation_per_regime(results["strategy_matrix"])

    if assets is not None and not assets.empty:
        results["asset_matrix"] = regime_performance_matrix(assets, regime, min_months)
        results["asset_sharpe_pivot"] = sharpe_pivot(results["asset_matrix"])
        if benchmark is not None and benchmark in assets.columns:
            results["asset_excess_vs_benchmark"] = excess_vs_benchmark(assets, regime, benchmark, min_months)

    if benchmark is not None and benchmark in strat.columns:
        results["strategy_excess_vs_benchmark"] = excess_vs_benchmark(strat, regime, benchmark, min_months)

    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        for name, df in results.items():
            df.to_csv(out / f"regime_{name}.csv", encoding="utf-8-sig")

    return results
