"""Validation tests for the pre-registered rebalancing policy.

Run:
  python tests\validation_rebalancing.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kq_tool.portfolio.rebalancing import RebalancePolicy, band_status, decide, turnover


def close(a: float, b: float, eps: float = 1e-9) -> bool:
    return abs(a - b) <= eps


def check(name: str, condition: bool) -> None:
    if not condition:
        raise AssertionError(name)
    print(f"[PASS] {name}")


def main() -> int:
    print("=" * 70)
    print("Rebalancing validation")
    print("=" * 70)
    policy = RebalancePolicy()

    target = {"stocks": 0.40, "bonds": 0.35, "gold": 0.15, "cash": 0.10}

    # R1. Inside all bands -> no trade.
    d = decide({"stocks": 0.42, "bonds": 0.34, "gold": 0.145, "cash": 0.095}, target, policy=policy)
    check("R1: inside bands holds", d.action == "hold" and d.trigger == "NO_TRADE")

    # R2. Absolute band breach on a large target triggers band rebalance.
    d = decide({"stocks": 0.50, "bonds": 0.28, "gold": 0.14, "cash": 0.08}, target, policy=policy)
    check("R2: absolute band breach triggers T1", d.action == "trade" and d.trigger == "T1_BAND_BREACH")

    # R3. Relative band matters for small target weights.
    rows = band_status({"cash": 0.13, "stocks": 0.37, "bonds": 0.35, "gold": 0.15}, target, policy)
    cash = next(row for row in rows if row["asset"] == "cash")
    check("R3: relative band detects small sleeve drift", cash["breached"] and close(cash["threshold"], 0.025))

    # R4. Official regime transition moves 70% toward the new target.
    current = {"stocks": 0.50, "bonds": 0.30, "gold": 0.10, "cash": 0.10}
    new_target = {"stocks": 0.30, "bonds": 0.35, "gold": 0.20, "cash": 0.15}
    d = decide(current, new_target, previous_regime="골디락스", current_regime="스태그플레이션", policy=policy)
    check("R4: regime transition uses 70pct partial move", close(d.target_weights["stocks"], 0.36))

    # R5. Re-evaluation flag alone never trades.
    d = decide(target, target, reeval_flag=True, policy=policy)
    check("R5: reeval flag alone is review-only", d.action == "hold" and d.turnover == 0.0 and "점검" in d.notes[0])

    # R6. Regime transition has priority over band breach.
    d = decide({"stocks": 0.60, "bonds": 0.20, "gold": 0.10, "cash": 0.10}, new_target,
               previous_regime="리플레이션", current_regime="스태그플레이션", policy=policy)
    check("R6: trigger priority chooses regime transition", d.trigger == "T2_REGIME_CHANGE")

    # R7. Partial band restoration trades less than full target rebalance.
    drifted = {"stocks": 0.50, "bonds": 0.28, "gold": 0.14, "cash": 0.08}
    d = decide(drifted, target, policy=policy)
    full_turnover = turnover(drifted, target)
    check("R7: partial restoration has lower turnover than full rebalance", 0 < d.turnover < full_turnover)

    print("-" * 70)
    print("Result: ALL PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
