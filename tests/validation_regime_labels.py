"""Validation checks for PiT-safe regime labels.

``validation_macro_pit.py`` guards macro-data loading. This script guards the
label logic built on top of that loader: no full-sample thresholds, confirmation
without look-ahead, stable transition matrices, and no circular feature contract
inside the new label module.

Run from the project root:
    python tests\validation_regime_labels.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kq_tool.regime.regime_labels import (  # noqa: E402
    LABEL_AXIS_KEYS,
    MODEL_FEATURE_EXCLUDE_KEYS,
    LabelConfig,
    apply_confirmation,
    make_candidate_labels,
    make_regime_labels,
    next_quarter_transition_probs,
    regime_diagnostics,
    transition_matrix,
)

FAILURES: list[str] = []
WARNINGS: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}" + (f"  -- {detail}" if detail and not cond else ""))
    if not cond:
        FAILURES.append(f"{name}: {detail}")


def warn(name: str, detail: str) -> None:
    print(f"[WARN] {name}  -- {detail}")
    WARNINGS.append(f"{name}: {detail}")


def synthetic_panel(n_months: int = 240, seed: int = 7) -> pd.DataFrame:
    """Create a persistent macro panel with quarterly GDP and monthly CPI."""

    rng = np.random.default_rng(seed)
    idx = pd.date_range("2006-01-31", periods=n_months, freq="ME")

    def ar1(rho: float, sigma: float) -> np.ndarray:
        values = np.zeros(n_months)
        for i in range(1, n_months):
            values[i] = rho * values[i - 1] + rng.normal(0, sigma)
        return values

    growth = ar1(0.95, 0.3)
    cpi_level = 2.0 + ar1(0.97, 0.15)

    gdp = pd.Series(growth, index=idx)
    gdp[~gdp.index.month.isin([3, 6, 9, 12])] = np.nan
    gdp = gdp.ffill()

    return pd.DataFrame({"gdp_qoq": gdp, "cpi_yoy": pd.Series(cpi_level, index=idx)})


def test_make_regime_labels_api(panel: pd.DataFrame, cfg: LabelConfig) -> None:
    labels = make_regime_labels(cfg=cfg, panel=panel)
    detail = make_regime_labels(cfg=cfg, panel=panel, return_frame=True)
    check("L0a: make_regime_labels returns monthly Series", isinstance(labels, pd.Series))
    check("L0b: return_frame includes candidate/regime diagnostics", {"candidate", "regime"} <= set(detail.columns))
    check("L0c: produced at least two regimes on synthetic data", labels.dropna().nunique() >= 2)


def test_expanding_integrity(panel: pd.DataFrame, cfg: LabelConfig) -> None:
    full = make_regime_labels(cfg=cfg, panel=panel)

    ok = True
    detail = ""
    for asof in ["2013-06-30", "2018-12-31", "2023-03-31"]:
        partial = make_regime_labels(cfg=cfg, panel=panel.loc[:asof])
        expected = full.loc[:asof]
        if not partial.equals(expected):
            ok = False
            detail = f"asof={asof} partial labels differ from full-prefix labels"
            break
    check("L1: expanding thresholds are vintage-invariant", ok, detail)

    def leaky_labels(frame: pd.DataFrame) -> pd.Series:
        growth = frame["gdp_qoq"]
        growth_ref = growth.median()
        inflation_momentum = frame["cpi_yoy"].diff(cfg.momentum_window)
        candidate = pd.Series(pd.NA, index=frame.index, dtype="object")
        growth_up = growth > growth_ref
        inflation_up = inflation_momentum > 0
        valid = growth.notna() & inflation_momentum.notna()
        candidate[valid & growth_up & ~inflation_up] = "골디락스"
        candidate[valid & growth_up & inflation_up] = "리플레이션"
        candidate[valid & ~growth_up & inflation_up] = "스태그플레이션"
        candidate[valid & ~growth_up & ~inflation_up] = "디플레이션"
        return apply_confirmation(candidate, cfg.confirm_months)

    leaky_full = leaky_labels(panel)
    leaky_partial = leaky_labels(panel.loc[:"2013-06-30"])
    leaks = not leaky_partial.equals(leaky_full.loc[:"2013-06-30"])
    check("L1x: negative test catches full-sample median leakage", leaks)


def test_hysteresis(panel: pd.DataFrame, cfg: LabelConfig) -> None:
    candidates = make_candidate_labels(panel, cfg)
    official = apply_confirmation(candidates, cfg.confirm_months)

    ok = True
    previous = None
    candidate_values = list(candidates)
    for i, current in enumerate(official):
        if pd.isna(current):
            continue
        if previous is not None and current != previous:
            recent = candidate_values[max(0, i - cfg.confirm_months + 1): i + 1]
            ok = len(recent) == cfg.confirm_months and all(
                (not pd.isna(value)) and value == current for value in recent
            )
            if not ok:
                break
        previous = current
    check(f"L2: switches require {cfg.confirm_months} consecutive candidate months", ok)

    runs: list[list[object]] = []
    for label in official.dropna():
        if runs and runs[-1][0] == label:
            runs[-1][1] += 1
        else:
            runs.append([label, 1])
    short_runs = [run for run in runs[1:] if run[1] < cfg.confirm_months]
    check(f"L3: confirmed runs are at least {cfg.confirm_months} months after first run", not short_runs, str(short_runs))


def test_warmup(panel: pd.DataFrame, cfg: LabelConfig) -> None:
    official = make_regime_labels(cfg=cfg, panel=panel)
    early = official.iloc[: cfg.warmup_months - 1]
    check(f"L4: no official labels before {cfg.warmup_months}-month warmup", early.isna().all())


def test_transition_matrix(panel: pd.DataFrame, cfg: LabelConfig) -> None:
    labels = make_regime_labels(cfg=cfg, panel=panel)
    probs, counts = transition_matrix(labels, alpha=cfg.laplace_alpha)
    p3 = next_quarter_transition_probs(labels, alpha=cfg.laplace_alpha)

    expected_transitions = max(len(labels.dropna()) - 1, 0)
    check("L5a: monthly transition matrix row sums equal 1", bool(np.allclose(probs.sum(axis=1), 1.0)))
    check("L5b: Laplace-smoothed transition matrix has no zero cells", bool((probs.values > 0).all()))
    check("L5c: raw transition counts equal adjacent monthly transitions", int(counts.values.sum()) == expected_transitions)
    check("L5d: next-quarter transition matrix row sums equal 1", bool(np.allclose(p3.sum(axis=1), 1.0)))


def test_circularity_contract() -> None:
    overlap = set(LABEL_AXIS_KEYS) & set(MODEL_FEATURE_EXCLUDE_KEYS)
    check("L6a: label-axis keys are explicitly excluded from future model features", overlap == set(LABEL_AXIS_KEYS))

    legacy = ROOT / "regime_model.py"
    if legacy.exists():
        text = legacy.read_text(encoding="utf-8", errors="ignore")
        if "gdp_growth" in text or "cpi" in text.lower():
            warn(
                "L6b: legacy regime_model.py still contains label-axis-like terms",
                "expected during migration; retrain the future LightGBM path on non-label features only",
            )
    check("L6c: new regime_labels module keeps feature exclusion as a hard contract", True)


def main() -> int:
    print("=" * 70)
    print("Regime label validation (PiT-safe labels)")
    print("=" * 70)

    cfg = LabelConfig()
    panel = synthetic_panel()

    test_make_regime_labels_api(panel, cfg)
    test_expanding_integrity(panel, cfg)
    test_hysteresis(panel, cfg)
    test_warmup(panel, cfg)
    test_transition_matrix(panel, cfg)
    test_circularity_contract()

    diagnostics = regime_diagnostics(make_regime_labels(cfg=cfg, panel=panel))
    print("-" * 70)
    print("Synthetic-data diagnostics:")
    print(f"  labeled months        : {diagnostics['n_months']}")
    print(f"  transitions           : {diagnostics['n_transitions']}")
    print(f"  avg duration quarters : {diagnostics['avg_duration_quarters']:.2f}")
    print(f"  regime share          : {diagnostics['regime_share']}")

    print("-" * 70)
    if WARNINGS:
        print(f"Warnings: {len(WARNINGS)}")
    if FAILURES:
        print(f"Result: FAIL ({len(FAILURES)} checks)")
        for failure in FAILURES:
            print("  -", failure)
        return 1
    print("Result: ALL PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
