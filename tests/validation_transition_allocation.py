"""Transition-Expected Allocation validation gate.

Run:
    python tests\validation_transition_allocation.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kq_tool.portfolio.recommender import REGIME_TARGETS  # noqa: E402
from kq_tool.portfolio.transition_allocation import (  # noqa: E402
    TEAConfig,
    compute_transition_expected_allocation,
    regime_weights_frame,
    transition_expected_regime_target,
)
from kq_tool.regime.ui_payload import build_payload  # noqa: E402

OUT = ROOT / "tests" / "analysis_outputs" / "transition_allocation_result.json"
FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  -- {detail}" if detail and not cond else ""))
    if not cond:
        FAILURES.append(name)


def demo_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    regimes = ["골디락스", "리플레이션", "스태그플레이션", "디플레이션"]
    matrix = pd.DataFrame(
        [
            [0.82, 0.12, 0.03, 0.03],
            [0.15, 0.62, 0.18, 0.05],
            [0.02, 0.10, 0.70, 0.18],
            [0.20, 0.05, 0.10, 0.65],
        ],
        index=regimes,
        columns=regimes,
    )
    weights = regime_weights_frame(REGIME_TARGETS).reindex(regimes)
    pi0 = pd.Series([0.0, 0.0, 1.0, 0.0], index=regimes)
    return matrix, weights, pi0


def main() -> int:
    print("=" * 72)
    print("Transition-Expected Allocation validation")
    print("=" * 72)
    matrix, weights, pi0 = demo_inputs()

    result = compute_transition_expected_allocation(matrix, weights, pi0, TEAConfig(horizon=3))
    check("T1: 최종 비중 합 == 1", np.isclose(result.weights.sum(), 1.0, atol=1e-8))
    check("T2: long-only 비중", bool((result.weights >= -1e-10).all()))

    lo = weights.min(axis=0)
    hi = weights.max(axis=0)
    check(
        "T3: 볼록결합 범위 내 비중",
        bool(((result.weights >= lo - 1e-8) & (result.weights <= hi + 1e-8)).all()),
    )

    sticky = pd.DataFrame(np.full(matrix.shape, 0.01 / 3), index=matrix.index, columns=matrix.columns)
    np.fill_diagonal(sticky.values, 0.99)
    sticky = sticky.div(sticky.sum(axis=1), axis=0)
    sticky_result = compute_transition_expected_allocation(sticky, weights, pi0, TEAConfig(horizon=3))
    check(
        "T4: 유지확률이 높으면 현재 배분 유지",
        sticky_result.alpha == 0.0
        and np.allclose(sticky_result.weights.values, sticky_result.weights_current.values),
    )

    long_result = compute_transition_expected_allocation(
        matrix,
        weights,
        pi0,
        TEAConfig(horizon=200, always_blend=True),
    )
    eigval, eigvec = np.linalg.eig(matrix.values.T)
    stat = np.real(eigvec[:, np.argmin(np.abs(eigval - 1.0))])
    stat = stat / stat.sum()
    stat = np.where(stat < 0, 0, stat)
    stat = stat / stat.sum()
    expected_stationary = pd.Series(stat @ weights.values, index=weights.columns)
    check(
        "T5: horizon 증가 시 정상분포 배분으로 수렴",
        np.allclose(long_result.weights_expected.values, expected_stationary.values, atol=1e-5),
    )

    rng = np.random.default_rng(42)
    diffs = []
    for _ in range(200):
        perm = rng.permutation(len(matrix))
        if np.array_equal(perm, np.arange(len(matrix))):
            continue
        shuffled = pd.DataFrame(matrix.values[perm][:, perm], index=matrix.index, columns=matrix.columns)
        shuffled_result = compute_transition_expected_allocation(
            shuffled,
            weights,
            pi0,
            TEAConfig(horizon=3, always_blend=True),
        )
        diffs.append(float((shuffled_result.weights_expected - result.weights_expected).abs().sum()))
    check("T6: 전이행렬 셔플 시 기대배분이 변함", float(np.median(diffs)) > 1e-3)

    payload = build_payload()
    snapshot = {
        "current": payload["current_regime"],
        "current_regime": payload["current_regime"],
        "probs": payload["nowcast_probs"],
        "next_quarter": payload["next_quarter_probs"],
        "transition_matrix": payload["transition_matrix"],
    }
    real_result = transition_expected_regime_target(snapshot, REGIME_TARGETS)
    check("T7: 실제 국면 payload와 REGIME_TARGETS 연결", np.isclose(real_result.weights.sum(), 1.0))

    out = {
        "status": "PASS" if not FAILURES else "FAIL",
        "demo": result.to_payload(),
        "real_payload": real_result.to_payload(),
        "shuffle_median_l1_shift": round(float(np.median(diffs)), 6),
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

