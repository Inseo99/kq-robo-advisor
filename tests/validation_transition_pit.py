"""Transition PiT gate for TEA.

Checks that TEA backtests estimate the regime transition matrix only from
labels observable at each historical month. This script is intentionally small
and deterministic so it can be included in the gate dashboard.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kq_tool.regime.transition_pit import (  # noqa: E402
    TransitionPitConfig,
    expanding_transition_matrices,
    full_sample_transition_matrix,
    load_regime_history,
    lookahead_distance_summary,
    posterior_transition_matrix,
    sticky_prior_matrix,
    transition_audit_payload,
    transition_counts,
)


def _synthetic_labels() -> pd.Series:
    dates = pd.date_range("2010-01-31", periods=144, freq="ME")
    regimes = (
        ["골디락스"] * 24
        + ["리플레이션"] * 18
        + ["스태그플레이션"] * 18
        + ["디플레이션"] * 24
        + ["골디락스"] * 18
        + ["리플레이션"] * 18
        + ["스태그플레이션"] * 24
    )
    return pd.Series(regimes, index=dates, name="regime")


def main() -> int:
    cfg = TransitionPitConfig(burn_in_months=60, n0=12, prior_sticky=0.85)
    labels = load_regime_history()

    print("=" * 72)
    print("Transition PiT Gate")
    print("=" * 72)
    print(f"real labels: {labels.index[0]:%Y-%m} ~ {labels.index[-1]:%Y-%m} ({len(labels)} months)")

    # G1: real label schema/coverage was validated by load_regime_history.
    assert len(labels) >= cfg.burn_in_months
    print("[G1] real labels loaded and monthly contiguous: PASS")

    # G2: expanding matrix is unchanged when future labels after t are modified.
    matrices = expanding_transition_matrices(labels, cfg)
    test_date = sorted(matrices)[len(matrices) // 2]
    polluted = labels.copy()
    polluted.loc[polluted.index > test_date] = "골디락스"
    polluted_matrix = expanding_transition_matrices(polluted, cfg)[test_date]
    assert np.allclose(matrices[test_date].values, polluted_matrix.values)
    print("[G2] future label pollution does not alter past P_t: PASS")

    # G3: final expanding P equals full-sample P.
    full = full_sample_transition_matrix(labels, cfg)
    final_date = max(matrices)
    assert np.allclose(matrices[final_date].values, full.values, atol=1e-12)
    print("[G3] final expanding P converges to full-sample P: PASS")

    # G4: Dirichlet shrinkage identity.
    small = _synthetic_labels().iloc[:72]
    counts = transition_counts(small)
    prior_sticky = 0.85
    prior = sticky_prior_matrix(sticky=prior_sticky)
    n0 = 12.0
    posterior = posterior_transition_matrix(counts, prior, n0)
    row = "골디락스"
    expected = (n0 * prior.loc[row] + counts.loc[row]) / (n0 + counts.loc[row].sum())
    assert np.allclose(posterior.loc[row].values, expected.values, atol=1e-15)
    print(f"[G4] Dirichlet shrinkage identity holds (sticky={prior_sticky}): PASS")

    # G5: quantify look-ahead impact; report as measurement, not forced tuning.
    summary = lookahead_distance_summary(labels, cfg)
    print(
        "[G5] look-ahead impact "
        f"early L1={summary['early_l1_mean']:.4f}, "
        f"late L1={summary['late_l1_mean']:.4f}, "
        f"max L1={summary['max_l1']:.4f}, "
        f"decayed={summary['decayed']}"
    )
    if not summary["decayed"]:
        print("     WARN: real transition events are clustered; report instead of tuning away.")

    payload = transition_audit_payload(labels, data_source="real", config=cfg)
    assert payload["status"] == "PASS"
    print("[G6] dashboard audit payload status: PASS")
    print("ALL PASS (실증 검증)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
