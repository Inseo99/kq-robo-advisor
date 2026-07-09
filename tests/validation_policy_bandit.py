"""Shadow-ledger policy bandit validation gate.

Run:
    python tests\validation_policy_bandit.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kq_tool.portfolio.policy_bandit import (  # noqa: E402
    BanditConfig,
    context_free_cum,
    run_shadow_bandit,
)

OUT = ROOT / "tests" / "analysis_outputs" / "policy_bandit_result.json"
N_PERM = 200
FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  -- {detail}" if detail and not cond else ""))
    if not cond:
        FAILURES.append(name)


def demo_inputs(n_periods: int = 180, seed: int = 11) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    regime_list = ["골디락스", "리플레이션", "스태그플레이션", "디플레이션"]
    transition = np.array(
        [
            [0.85, 0.10, 0.02, 0.03],
            [0.10, 0.70, 0.15, 0.05],
            [0.02, 0.08, 0.75, 0.15],
            [0.20, 0.05, 0.05, 0.70],
        ]
    )
    states = [0]
    for _ in range(n_periods - 1):
        states.append(int(rng.choice(4, p=transition[states[-1]])))
    dates = pd.date_range("2011-01-31", periods=n_periods, freq="ME")
    regimes = pd.Series([regime_list[state] for state in states], index=dates)

    policies = ["band", "quarterly", "never"]
    mean_by_regime = {
        "골디락스": {"band": 0.010, "quarterly": 0.011, "never": 0.014},
        "리플레이션": {"band": 0.006, "quarterly": 0.004, "never": 0.002},
        "스태그플레이션": {"band": 0.000, "quarterly": -0.006, "never": -0.012},
        "디플레이션": {"band": 0.012, "quarterly": 0.013, "never": 0.010},
    }
    rows = np.zeros((n_periods, len(policies)))
    for i, regime in enumerate(regimes.values):
        common_shock = rng.normal(0, 0.03)
        for j, policy in enumerate(policies):
            rows[i, j] = mean_by_regime[regime][policy] + common_shock + rng.normal(0, 0.008)
    rewards = pd.DataFrame(rows, index=dates, columns=policies)
    return rewards, regimes


def load_real_inputs() -> tuple[pd.DataFrame, pd.Series]:
    ledger_path = ROOT / "data" / "state" / "shadow_ledger.csv"
    labels_path = ROOT / "data" / "macro" / "regime_labels.csv"
    ledger = pd.read_csv(ledger_path, index_col=0, parse_dates=True)
    nav_cols = ["band_nav", "quarterly_nav", "never_nav"]
    rewards = ledger[nav_cols].pct_change().dropna()
    rewards.columns = ["band", "quarterly", "never"]

    labels = pd.read_csv(labels_path, parse_dates=["date"])
    labels = labels.set_index("date")["regime"].astype(str)
    labels.index = labels.index.to_period("M").to_timestamp("M")
    return rewards, labels


def main() -> int:
    print("=" * 72)
    print("Shadow Policy Bandit validation")
    print("=" * 72)
    rewards, regimes = demo_inputs()
    cfg = BanditConfig()
    result = run_shadow_bandit(rewards, regimes, cfg)

    midpoint = rewards.index[len(rewards) // 2]
    poisoned = rewards.copy()
    poisoned.loc[poisoned.index > midpoint] = 9.99
    poisoned_result = run_shadow_bandit(poisoned, regimes, cfg)
    check("B1: 미래 보상 오염에도 과거 확률 동일", result.probs.loc[:midpoint].equals(poisoned_result.probs.loc[:midpoint]))

    uniform_cum = float(rewards.mean(axis=1).sum())
    best_cum = float(result.static_cum.max())
    recovery = (result.bandit_cum - uniform_cum) / (best_cum - uniform_cum) if best_cum > uniform_cum else 1.0
    check("B2: 균등 정책 혼합 대비 우수", result.bandit_cum > uniform_cum)
    check("B3: best static 대비 gap 일부 회수", recovery >= 0.50, f"recovery={recovery:.3f}")

    rng = np.random.default_rng(2026)
    context_free = context_free_cum(rewards, cfg)
    real_advantage = result.bandit_cum - context_free
    perm_advantages = []
    for _ in range(N_PERM):
        shuffled_regimes = pd.Series(rng.permutation(regimes.values), index=regimes.index)
        shuffled_result = run_shadow_bandit(rewards, shuffled_regimes, cfg)
        perm_advantages.append(shuffled_result.bandit_cum - context_free)
    p_regime = float(np.mean([adv >= real_advantage for adv in perm_advantages]))
    check("B4: 국면 permutation에서 우위가 유의", p_regime <= 0.10, f"p={p_regime:.3f}")

    perm_edges = []
    real_edge = result.bandit_cum - uniform_cum
    for _ in range(N_PERM):
        shuffled = rewards.copy()
        for col in shuffled.columns:
            values = shuffled[col].values.copy()
            rng.shuffle(values)
            shuffled[col] = values
        shuffled_result = run_shadow_bandit(shuffled, regimes, cfg)
        perm_edges.append(shuffled_result.bandit_cum - float(shuffled.mean(axis=1).sum()))
    p_reward = float(np.mean([edge >= real_edge for edge in perm_edges]))
    check("B5: 보상 permutation에서 우위가 유의", p_reward <= 0.10, f"p={p_reward:.3f}")

    short_result = run_shadow_bandit(rewards.iloc[:18], regimes.iloc[:18], cfg)
    rare_regime = short_result.n_by_regime.idxmin()
    rare_n = int(short_result.n_by_regime.min())
    rare_probs = short_result.regime_weights_final.loc[rare_regime].values
    tv_from_uniform = 0.5 * float(np.abs(rare_probs - 1.0 / len(rare_probs)).sum())
    kappa = rare_n / (rare_n + cfg.n0)
    check("B6: 소표본 국면은 shrinkage로 과감한 선택 억제", rare_n <= cfg.n0 and tv_from_uniform <= max(0.35, kappa + 0.20))
    check("B7: shadow_only 플래그 명시", result.to_payload().get("shadow_only") is True)

    real_payload: dict[str, object] = {}
    try:
        real_rewards, real_regimes = load_real_inputs()
        real_result = run_shadow_bandit(real_rewards, real_regimes, cfg)
        real_payload = real_result.to_payload()
        check("B8: 실제 섀도 원장 + 국면 라벨 스모크", len(real_result.meta_returns) >= 24)
    except Exception as exc:  # noqa: BLE001
        check("B8: 실제 섀도 원장 + 국면 라벨 스모크", False, str(exc))

    out = {
        "status": "PASS" if not FAILURES else "FAIL",
        "synthetic": {
            "bandit_cum": round(float(result.bandit_cum), 6),
            "uniform_cum": round(uniform_cum, 6),
            "best_static": result.best_static,
            "gap_recovery_ratio": round(float(recovery), 4),
            "regime_perm_pvalue": round(p_regime, 4),
            "reward_perm_pvalue": round(p_reward, 4),
            "rare_regime": str(rare_regime),
            "rare_regime_n": rare_n,
            "rare_tv_from_uniform": round(tv_from_uniform, 4),
            "payload": result.to_payload(),
        },
        "real_smoke": real_payload,
        "n_permutations": N_PERM,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print("-" * 72)
    if FAILURES:
        print(f"결과: FAIL ({len(FAILURES)}건)")
        return 1
    print(f"결과: ALL PASS - 저장: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

