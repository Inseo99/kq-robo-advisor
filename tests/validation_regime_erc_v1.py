"""Validation gate for regime ERC v1 allocation table."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kq_tool.portfolio.regime_erc import (  # noqa: E402
    DEFAULT_ASSET_BOUNDS,
    DEFAULT_REGIME_ASSET_BOUNDS,
    RegimeERCConfig,
    build_regime_erc_allocations,
    shrink_covariance,
)
from kq_tool.regime.regime_labels import REGIMES  # noqa: E402
from kq_tool.regime.transition_pit import load_regime_history  # noqa: E402


def _load_returns() -> pd.DataFrame:
    path = ROOT / "data" / "assets" / "monthly_returns.csv"
    df = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
    df.index = df.index.to_period("M").to_timestamp("M")
    return df[["stocks", "bonds", "gold", "cash"]].astype(float)


def main() -> int:
    returns = _load_returns()
    labels = load_regime_history()
    cfg = RegimeERCConfig(n0=36.0)
    weights, audit = build_regime_erc_allocations(returns, labels, cfg)

    print("=" * 72)
    print("Regime ERC v1 Gate")
    print("=" * 72)

    assert list(weights.index) == list(REGIMES)
    assert np.allclose(weights.sum(axis=1).values, 1.0, atol=1e-8)
    print("[E1] one row per regime and row sums are 1: PASS")

    for regime, bounds in DEFAULT_REGIME_ASSET_BOUNDS.items():
        for asset, (lo, hi) in bounds.items():
            assert weights.loc[regime, asset] >= lo - 1e-8
            assert weights.loc[regime, asset] <= hi + 1e-8
    print("[E2] fixed asset bounds respected: PASS")

    common = returns.index.intersection(labels.index)
    aligned_returns = returns.loc[common]
    aligned_labels = labels.loc[common]
    for regime in REGIMES:
        subset = aligned_returns[aligned_labels == regime]
        _, kappa = shrink_covariance(subset, aligned_returns, cfg.n0, cfg.ridge)
        reported = float(audit.loc[audit["regime"].eq(regime), "kappa"].iloc[0])
        assert abs(kappa - reported) < 1e-12
    print("[E3] kappa = n/(n+n0) audit identity: PASS")

    assert (audit["n_months"] >= 1).all()
    assert ((audit["kappa"] > 0) & (audit["kappa"] < 1)).all()
    print("[E4] real regime samples use shrinkage, not pure in-sample covariance: PASS")

    out_dir = ROOT / "tests" / "analysis_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    weights.to_csv(out_dir / "regime_erc_v1_weights_test.csv", encoding="utf-8-sig")
    audit.to_csv(out_dir / "regime_erc_v1_audit_test.csv", index=False, encoding="utf-8-sig")
    print((weights * 100).round(1).to_string())
    print("ALL PASS (실증 검증)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
