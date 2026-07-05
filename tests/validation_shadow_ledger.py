"""Shadow ledger validation.

Run:
  python tests\validation_shadow_ledger.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kq_tool.portfolio.shadow_ledger import DEFAULT_TARGET, MODES, build_ledger, ledger_summary

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}" + (f" -- {detail}" if detail and not condition else ""))
    if not condition:
        FAILURES.append(name)


def synth_returns(n: int = 120, seed: int = 4) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2016-01-31", periods=n, freq="ME")
    return pd.DataFrame(
        {
            "stocks": rng.normal(0.007, 0.050, n),
            "bonds": rng.normal(0.002, 0.012, n),
            "gold": rng.normal(0.003, 0.035, n),
            "cash": rng.normal(0.002, 0.001, n),
        },
        index=idx,
    )


def main() -> int:
    print("=" * 70)
    print("Shadow ledger validation")
    print("=" * 70)
    returns = synth_returns()

    # S1. Deterministic idempotent rebuild.
    ledger_a = build_ledger(returns)
    ledger_b = build_ledger(returns)
    check("S1: identical input produces identical ledger", ledger_a.equals(ledger_b))

    # S2. Vintage invariance: removing future data leaves past ledger unchanged.
    cut = build_ledger(returns.iloc[:-24])
    check("S2: vintage invariance / no look-ahead", ledger_a.iloc[:-24].equals(cut))

    # S3. Accounting integrity.
    check(
        "S3a: never mode has zero turnover and cost",
        float(ledger_a["never_turnover"].sum()) == 0.0 and float(ledger_a["never_cost"].sum()) == 0.0,
    )
    check(
        "S3b: cost equals turnover times bps",
        bool(np.allclose(ledger_a["band_cost"], ledger_a["band_turnover"] * 15.0 / 10_000.0)),
    )
    check(
        "S3c: all NAV values stay positive",
        bool((ledger_a[[f"{mode}_nav" for mode in MODES]] > 0).all().all()),
    )

    # S4. Quarterly control trades only at 3-month cadence.
    q_turn = ledger_a["quarterly_turnover"]
    nonzero_positions = [i + 1 for i, value in enumerate(q_turn) if value > 0]
    check("S4: quarterly trades only every 3 months", all(pos % 3 == 0 for pos in nonzero_positions))

    # S5. Summary identities.
    summary = ledger_summary(ledger_a)
    saved = summary["quarterly"]["cum_cost_pctp"] - summary["band"]["cum_cost_pctp"]
    check(
        "S5a: cost_saved identity holds",
        abs(summary["band_vs_quarterly"]["cost_saved_pctp"] - round(saved, 4)) < 1e-9,
    )
    check(
        "S5b: summary contains pre-registered verdict",
        summary["band_vs_quarterly"]["verdict"] in {
            "밴드 유지",
            "정책 재검토 트리거 - 분기 캘린더가 허용오차 밖에서 우월",
        },
    )
    check("S5c: target weights sum to 1", abs(sum(DEFAULT_TARGET.values()) - 1.0) < 1e-12)

    print("-" * 70)
    if FAILURES:
        print(f"Result: FAIL ({len(FAILURES)} checks)")
        return 1
    print("Result: ALL PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
