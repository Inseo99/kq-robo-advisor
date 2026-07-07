"""Validation script for deflated_sharpe and regime_performance.

Run:
  python tests/validation_dsr_regime.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from kq_tool.validation.deflated_sharpe import (  # noqa: E402
    bootstrap_sharpe_ci,
    deflated_sharpe_ratio,
    deflated_sharpe_table,
    expected_max_sharpe,
    probabilistic_sharpe_ratio,
    sharpe_ratio,
)
from kq_tool.validation.regime_performance import (  # noqa: E402
    best_allocation_per_regime,
    excess_vs_benchmark,
    regime_performance_matrix,
    sharpe_pivot,
)

RNG = np.random.default_rng(7)
PASS: list[str] = []
FAIL: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    (PASS if condition else FAIL).append(name)
    print(f"  {'PASS' if condition else 'FAIL'} - {name}" + (f" ({detail})" if detail else ""))


print("[A] expected_max_sharpe monotonicity")
variance = 0.04
e1 = expected_max_sharpe(1, variance)
e5 = expected_max_sharpe(5, variance)
e15 = expected_max_sharpe(15, variance)
e100 = expected_max_sharpe(100, variance)
check("N=1 returns zero benchmark", abs(e1) < 1e-12)
check("expected max SR increases with N", e1 < e5 < e15 < e100)

print("[B] PSR behavior")
t = 240
good = RNG.normal(0.01, 0.03, t)
noise = RNG.normal(0.0, 0.03, t)
psr_good = probabilistic_sharpe_ratio(good, 0.0)
psr_noise = probabilistic_sharpe_ratio(noise, 0.0)
check("positive-alpha PSR > 0.99", psr_good > 0.99, f"{psr_good:.4f}")
check("noise PSR not extreme", 0.01 < psr_noise < 0.99, f"{psr_noise:.4f}")

print("[C] DSR deflates best-of-15 noise")
n_trials, reps, false_pos = 15, 60, 0
for _ in range(reps):
    trials = RNG.normal(0.0, 0.03, size=(n_trials, 120))
    srs = [sharpe_ratio(trial) for trial in trials]
    best = trials[int(np.argmax(srs))]
    result = deflated_sharpe_ratio(best, srs, n_trials=n_trials)
    if result["significant_5pct"]:
        false_pos += 1
fp_rate = false_pos / reps
check("best-of-15 noise DSR false positive rate <= 10%", fp_rate <= 0.10, f"{fp_rate:.2%}")

naive_pos = 0
for _ in range(reps):
    trials = RNG.normal(0.0, 0.03, size=(n_trials, 120))
    srs = [sharpe_ratio(trial) for trial in trials]
    best = trials[int(np.argmax(srs))]
    if probabilistic_sharpe_ratio(best, 0.0) >= 0.95:
        naive_pos += 1
print(f"  참고: 보정 없는 PSR>=0.95 오탐률 {naive_pos / reps:.2%}")

print("[D] True alpha survives DSR")
alpha = RNG.normal(0.012, 0.03, 240)
noise_pool = RNG.normal(0.0, 0.03, size=(14, 240))
srs = [sharpe_ratio(alpha)] + [sharpe_ratio(trial) for trial in noise_pool]
result = deflated_sharpe_ratio(alpha, srs, n_trials=15)
check("alpha DSR >= 0.95", result["dsr"] >= 0.95, f"DSR={result['dsr']:.4f}")

print("[E] table and bootstrap CI smoke")
idx = pd.period_range("2005-01", periods=240, freq="M").to_timestamp("M")
df = pd.DataFrame(
    {"alpha_strat": alpha, "noise1": noise_pool[0], "noise2": noise_pool[1]},
    index=idx,
)
table = deflated_sharpe_table(df, n_trials=15)
check("table sorted with alpha first", table.shape[0] == 3 and table.index[0] == "alpha_strat")
ci = bootstrap_sharpe_ci(df["alpha_strat"], n_boot=2_000)
check(
    "CI contains point estimate",
    ci["ci_low"] <= ci["sharpe_annualized"] <= ci["ci_high"],
    f"{ci['sharpe_annualized']:.3f} [{ci['ci_low']:.3f}, {ci['ci_high']:.3f}]",
)

print("[F] regime decomposition detects planted regime effect")
regimes = pd.Series(
    RNG.choice(["goldilocks", "reflation", "deflation", "stagflation"], size=240, p=[0.35, 0.25, 0.2, 0.2]),
    index=idx,
    name="regime",
)
bench = pd.Series(RNG.normal(0.004, 0.04, 240), index=idx)
strategy = bench + RNG.normal(0.0, 0.01, 240)
strategy[regimes == "goldilocks"] += 0.015
returns = pd.DataFrame({"regime_strat": strategy, "BENCH": bench})
returns.index = returns.index.to_period("M")
regimes.index = regimes.index.to_period("M")

matrix = regime_performance_matrix(returns, regimes, min_months=12)
pivot = sharpe_pivot(matrix)
gl = pivot.loc["goldilocks", "regime_strat"]
other = pivot.drop("goldilocks")["regime_strat"].mean()
check("goldilocks Sharpe materially higher", gl > other + 0.5, f"{gl:.2f} vs {other:.2f}")

excess = excess_vs_benchmark(returns, regimes, benchmark="BENCH", min_months=12)
gl_row = excess[(excess.regime == "goldilocks") & (excess.strategy == "regime_strat")].iloc[0]
check("goldilocks excess is significant", gl_row["p_value"] < 0.05, f"p={gl_row['p_value']:.4f}")
non_gl = excess[(excess.regime != "goldilocks") & (excess.strategy == "regime_strat")]
check("other regimes mostly not significant", (non_gl["p_value"] > 0.05).mean() >= 2 / 3)

best = best_allocation_per_regime(matrix)
check(
    "best-per-regime table generated",
    not best.empty and best.loc[best.regime == "goldilocks", "strategy"].iloc[0] == "regime_strat",
)

print()
print(f"결과: {len(PASS)} PASS / {len(FAIL)} FAIL")
if FAIL:
    print("실패:", FAIL)
    raise SystemExit(1)
