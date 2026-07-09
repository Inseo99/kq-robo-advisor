"""TEA target smoothness and band-frequency gate.

This gate checks the separation we agreed on:

- TEA target weights are a stateless function of current regime belief and P_t.
- Turnover control remains in the rebalancing layer.

The test uses real regime labels and real asset-class monthly returns. It does
not tune band widths; it reports 3/5/7%p sensitivity for the dashboard.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kq_tool.portfolio.rebalancing import RebalancePolicy, decide, normalize_weights  # noqa: E402
from kq_tool.portfolio.transition_allocation import (  # noqa: E402
    TEAConfig,
    compute_transition_expected_allocation,
    regime_weights_frame,
)
from kq_tool.regime.regime_labels import REGIMES  # noqa: E402
from kq_tool.regime.transition_pit import (  # noqa: E402
    TransitionPitConfig,
    expanding_transition_matrices,
    load_regime_history,
)


ASSET_REGIME_TARGETS = {
    "골디락스": {"stocks": 0.65, "bonds": 0.15, "gold": 0.05, "cash": 0.15},
    "리플레이션": {"stocks": 0.45, "bonds": 0.10, "gold": 0.20, "cash": 0.25},
    "스태그플레이션": {"stocks": 0.20, "bonds": 0.15, "gold": 0.35, "cash": 0.30},
    "디플레이션": {"stocks": 0.15, "bonds": 0.55, "gold": 0.10, "cash": 0.20},
}


def _load_returns() -> pd.DataFrame:
    path = ROOT / "data" / "assets" / "monthly_returns.csv"
    df = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
    df.index = df.index.to_period("M").to_timestamp("M")
    return df[["stocks", "bonds", "gold", "cash"]].astype(float)


def _one_hot(regime: str) -> pd.Series:
    s = pd.Series(0.0, index=REGIMES)
    s.loc[regime] = 1.0
    return s


def _target_paths(labels: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    pit_cfg = TransitionPitConfig(burn_in_months=60, n0=12, prior_sticky=0.85)
    tea_cfg = TEAConfig(horizon=3, p_stay_lo=0.50, p_stay_hi=0.80)
    matrices = expanding_transition_matrices(labels, pit_cfg)
    table = regime_weights_frame(ASSET_REGIME_TARGETS)
    step_rows = []
    tea_rows = []
    for date, matrix in matrices.items():
        regime = str(labels.loc[date])
        step = table.loc[regime]
        tea = compute_transition_expected_allocation(
            matrix,
            table,
            _one_hot(regime),
            tea_cfg,
        ).weights
        step_rows.append((date, step))
        tea_rows.append((date, tea))
    step_df = pd.DataFrame({d: row for d, row in step_rows}).T.sort_index()
    tea_df = pd.DataFrame({d: row for d, row in tea_rows}).T.sort_index()
    return step_df, tea_df


def _mean_l1_change(targets: pd.DataFrame) -> float:
    return float(targets.diff().abs().sum(axis=1).dropna().mean())


def _max_l1_change(targets: pd.DataFrame) -> float:
    return float(targets.diff().abs().sum(axis=1).dropna().max())


def _apply_return(weights: dict[str, float], ret: pd.Series) -> dict[str, float]:
    raw = {asset: float(weights.get(asset, 0.0)) * (1.0 + float(ret.get(asset, 0.0))) for asset in ret.index}
    total = sum(max(v, 0.0) for v in raw.values())
    if total <= 0:
        return normalize_weights(weights)
    return {asset: max(v, 0.0) / total for asset, v in raw.items()}


def _simulate_band(
    targets: pd.DataFrame,
    returns: pd.DataFrame,
    abs_band: float,
) -> dict[str, float]:
    common = targets.index.intersection(returns.index)
    if len(common) < 24:
        raise RuntimeError("Not enough common target/return months")
    policy = RebalancePolicy(abs_band=abs_band, rel_band=0.25)
    current = normalize_weights(targets.loc[common[0]].to_dict())
    n_trade = 0
    total_turnover = 0.0
    for date in common[1:]:
        current = _apply_return(current, returns.loc[date])
        target = targets.loc[date].to_dict()
        decision = decide(current, target, policy=policy)
        if decision.action == "trade":
            n_trade += 1
            total_turnover += float(decision.turnover)
            current = normalize_weights(decision.target_weights)
    years = max(len(common) / 12.0, 1e-9)
    return {
        "months": float(len(common)),
        "trades": float(n_trade),
        "trades_per_year": float(n_trade / years),
        "turnover": float(total_turnover),
        "turnover_per_year": float(total_turnover / years),
    }


def main() -> int:
    labels = load_regime_history()
    returns = _load_returns()
    step, tea = _target_paths(labels)
    common = step.index.intersection(tea.index).intersection(returns.index)
    step = step.loc[common]
    tea = tea.loc[common]

    print("=" * 72)
    print("TEA Band Gate")
    print("=" * 72)
    print(f"period: {common[0]:%Y-%m} ~ {common[-1]:%Y-%m} ({len(common)} months)")

    step_mean = _mean_l1_change(step)
    tea_mean = _mean_l1_change(tea)
    step_max = _max_l1_change(step)
    tea_max = _max_l1_change(tea)
    print(f"[B1] mean monthly target L1: step={step_mean:.4f}, TEA={tea_mean:.4f}")
    print(f"[B2] max monthly target L1 : step={step_max:.4f}, TEA={tea_max:.4f}")
    assert tea_mean <= step_mean + 1e-12
    assert tea_max <= step_max + 1e-12
    print("[B1/B2] TEA target path is smoother than step target: PASS")

    rows = []
    for band in [0.03, 0.05, 0.07]:
        step_stats = _simulate_band(step, returns, band)
        tea_stats = _simulate_band(tea, returns, band)
        row = {
            "abs_band": band,
            "step_trades": step_stats["trades"],
            "tea_trades": tea_stats["trades"],
            "step_turnover": step_stats["turnover"],
            "tea_turnover": tea_stats["turnover"],
            "step_trades_per_year": step_stats["trades_per_year"],
            "tea_trades_per_year": tea_stats["trades_per_year"],
        }
        rows.append(row)
        print(
            f"[B3] band {band:.0%}p: "
            f"step trades={row['step_trades']:.0f}, TEA trades={row['tea_trades']:.0f}, "
            f"step turnover={row['step_turnover']:.3f}, TEA turnover={row['tea_turnover']:.3f}"
        )

    sens = pd.DataFrame(rows)
    base = sens[sens["abs_band"].eq(0.05)].iloc[0]
    assert base["tea_trades"] <= base["step_trades"] + 3
    print("[B4] 5%p band trade frequency not materially increased: PASS")

    out_dir = ROOT / "tests" / "analysis_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    sens.to_csv(out_dir / "tea_band_sensitivity.csv", index=False)
    pd.concat({"step": step, "tea": tea}, axis=1).to_csv(out_dir / "tea_target_paths.csv")
    print(f"saved: {out_dir / 'tea_band_sensitivity.csv'}")
    print("ALL PASS (실증 검증)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
