"""Validation checks for regime_model_v2.

The label validation script catches leakage in target construction. This script
checks the model path: no circular features, purged walk-forward predictions,
probability quality, and baseline comparison.

Run from project root:
    python tests\validation_regime_model.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kq_tool.regime.regime_labels import LabelConfig, apply_confirmation, make_candidate_labels  # noqa: E402
from kq_tool.regime.regime_model_v2 import (  # noqa: E402
    FEATURES,
    FORBIDDEN_FEATURE_TERMS,
    ModelConfig,
    build_features,
    calibration_table,
    log_loss_score,
    naive_nowcast_baseline,
    p3_forecast_baseline,
    walk_forward_predict,
)

FAILURES: list[str] = []
WARNINGS: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}" + (f"  -- {detail}" if detail and not cond else ""))
    if not cond:
        FAILURES.append(f"{name}: {detail}")


def info(name: str, msg: str) -> None:
    print(f"[INFO] {name}: {msg}")


def synthetic_world(n_months: int = 300, seed: int = 11) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2001-01-31", periods=n_months, freq="ME")

    def ar1(rho: float, sigma: float) -> np.ndarray:
        values = np.zeros(n_months)
        for i in range(1, n_months):
            values[i] = rho * values[i - 1] + rng.normal(0, sigma)
        return values

    growth = ar1(0.96, 0.25)
    inflation_momentum = ar1(0.97, 0.12)

    gdp = pd.Series(growth, index=idx)
    gdp[~gdp.index.month.isin([3, 6, 9, 12])] = np.nan
    gdp = gdp.ffill() + rng.normal(0, 0.05, n_months)
    cpi = pd.Series(2.0 + np.cumsum(inflation_momentum) * 0.05, index=idx)
    label_panel = pd.DataFrame({"gdp_qoq": gdp, "cpi_yoy": cpi})

    growth_lead = pd.Series(growth, index=idx).shift(-2).ffill()
    inflation_lead = pd.Series(inflation_momentum, index=idx).shift(-2).ffill()
    market = pd.DataFrame(index=idx)
    market["yield_spread_10y_3y"] = 1.0 + 0.8 * growth_lead - 0.3 * inflation_lead + rng.normal(0, 0.15, n_months)
    market["credit_spread"] = 0.8 - 0.6 * growth_lead + rng.normal(0, 0.1, n_months)
    market["usdkrw"] = 1200 * np.exp(np.cumsum(-0.004 * growth_lead.to_numpy() + rng.normal(0, 0.01, n_months)))
    market["kospi"] = 2000 * np.exp(
        np.cumsum(0.008 * growth_lead.to_numpy() - 0.003 * inflation_lead.to_numpy() + rng.normal(0, 0.03, n_months))
    )
    market["base_rate"] = 2.0 + 0.5 * pd.Series(inflation_momentum, index=idx).rolling(6).mean().fillna(0.0)
    return label_panel, market


def test_m1() -> None:
    bad = [feature for feature in FEATURES if any(term in feature.lower() for term in FORBIDDEN_FEATURE_TERMS)]
    check("M1a: FEATURES exclude label-axis terms", not bad, f"forbidden features: {bad}")

    src = (ROOT / "src" / "kq_tool" / "regime" / "regime_model_v2.py").read_text(encoding="utf-8")
    feature_block = src.split("FEATURES = [", 1)[1].split("]", 1)[0]
    bad_terms = [term for term in FORBIDDEN_FEATURE_TERMS if term in feature_block.lower()]
    check("M1b: static scan finds no forbidden terms in feature block", not bad_terms, str(bad_terms))


def test_m2(features: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
    cfg = ModelConfig(task="nowcast")
    probs, audit = walk_forward_predict(features, labels, cfg)
    check("M2a: walk-forward produced probabilities", not probs.empty)
    check("M2b: audit produced train-set metadata", not audit.empty)

    ok_audit = bool((audit["train_max_settle"] <= audit.index).all()) if not audit.empty else False
    check("M2c: every prediction uses only labels settled by T", ok_audit)

    cut = features.index[-48]
    probs_cut, _ = walk_forward_predict(features.loc[:cut], labels.loc[:cut], cfg)
    common = probs.index.intersection(probs_cut.index)
    same = len(common) > 0 and np.allclose(
        probs.loc[common].to_numpy(dtype=float),
        probs_cut.loc[common].to_numpy(dtype=float),
        atol=1e-9,
    )
    check("M2d: vintage invariance after removing future data", same)
    return probs


def test_m3(probs: pd.DataFrame, labels: pd.Series) -> None:
    row_sums_ok = bool(np.allclose(probs.sum(axis=1), 1.0, atol=1e-6))
    check("M3a: probability row sums equal 1", row_sums_ok)

    mean_max = float(probs.max(axis=1).mean())
    check("M3b: probabilities are not degenerate 100/0 style", mean_max < 0.99, f"mean max={mean_max:.4f}")
    info("M3", f"mean max probability = {mean_max:.3f}")

    table = calibration_table(probs, labels)
    print("      Calibration by confidence bucket:")
    for bucket, row in table.iterrows():
        print(f"        {bucket}: n={int(row['n'])}, conf={row['avg_conf']:.2f}, hit={row['hit_rate']:.2f}")


def test_m4(features: pd.DataFrame, labels: pd.Series) -> None:
    cfg_now = ModelConfig(task="nowcast")
    probs_now, _ = walk_forward_predict(features, labels, cfg_now)
    naive = naive_nowcast_baseline(labels, cfg_now)
    common_now = probs_now.index.intersection(naive.index)
    ll_model = log_loss_score(probs_now.loc[common_now], labels)
    ll_naive = log_loss_score(naive.loc[common_now], labels)
    info("M4-nowcast", f"model {ll_model:.4f} vs naive {ll_naive:.4f}")
    check("M4a: nowcast log-loss values are finite", np.isfinite(ll_model) and np.isfinite(ll_naive))
    if np.isfinite(ll_model) and np.isfinite(ll_naive) and ll_model >= ll_naive:
        WARNINGS.append("nowcast model did not beat naive persistence; keep baseline until real data says otherwise")

    cfg_fore = ModelConfig(task="forecast", horizon_m=3)
    probs_fore, _ = walk_forward_predict(features, labels, cfg_fore)
    actual_fore = labels.shift(-cfg_fore.horizon_m)
    p3 = p3_forecast_baseline(labels, cfg_fore)
    common_fore = probs_fore.index.intersection(p3.index)
    ll_model_f = log_loss_score(probs_fore.loc[common_fore], actual_fore)
    ll_p3 = log_loss_score(p3.loc[common_fore], actual_fore)
    info("M4-forecast", f"model {ll_model_f:.4f} vs P^3 {ll_p3:.4f}")
    check("M4b: forecast log-loss values are finite", np.isfinite(ll_model_f) and np.isfinite(ll_p3))
    if np.isfinite(ll_model_f) and np.isfinite(ll_p3) and ll_model_f >= ll_p3:
        WARNINGS.append("forecast model did not beat P^3 baseline; keep transition baseline until real data says otherwise")


def main() -> int:
    print("=" * 70)
    print("Regime model v2 validation")
    print("=" * 70)
    try:
        import lightgbm  # noqa: F401
        print("Backend: LightGBM")
    except ImportError:
        print("Backend: sklearn HistGradientBoosting fallback")

    label_panel, market = synthetic_world()
    label_cfg = LabelConfig()
    labels = apply_confirmation(make_candidate_labels(label_panel, label_cfg), label_cfg.confirm_months)
    features = build_features(market)

    test_m1()
    probs = test_m2(features, labels)
    test_m3(probs, labels)
    test_m4(features, labels)

    print("-" * 70)
    for warning in WARNINGS:
        print("  [주의]", warning)
    if FAILURES:
        print(f"Result: FAIL ({len(FAILURES)} checks)")
        for failure in FAILURES:
            print("  -", failure)
        return 1
    print("Result: ALL PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
