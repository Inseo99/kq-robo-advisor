"""Regime label vintage-stability gate.

The macro PiT gate checks whether raw data are observable. This gate checks the
next layer: whether the labeler itself rewrites past labels when future macro
data are appended. It also includes a negative-control labeler that uses
full-sample statistics, proving the gate can catch a look-ahead-prone method.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kq_tool.regime.macro_data import get_observable_panel  # noqa: E402
from kq_tool.regime.regime_labels import REGIMES, LabelConfig, make_regime_labels  # noqa: E402
from kq_tool.regime.transition_pit import load_regime_history  # noqa: E402


def _agreement(a: pd.Series, b: pd.Series) -> float:
    common = a.dropna().index.intersection(b.dropna().index)
    if len(common) == 0:
        return float("nan")
    return float((a.loc[common].astype(str) == b.loc[common].astype(str)).mean())


def _cohen_kappa(a: pd.Series, b: pd.Series) -> float:
    common = a.dropna().index.intersection(b.dropna().index)
    if len(common) == 0:
        return float("nan")
    x = a.loc[common].astype(str)
    y = b.loc[common].astype(str)
    observed = float((x == y).mean())
    expected = 0.0
    for regime in REGIMES:
        expected += float((x == regime).mean()) * float((y == regime).mean())
    denom = 1.0 - expected
    return 1.0 if abs(denom) < 1e-12 else (observed - expected) / denom


def _bad_full_sample_labeler(asof: str | None = None) -> pd.Series:
    """Intentionally unsafe labeler: z-scores use each vintage's full sample."""

    panel = get_observable_panel(["gdp_qoq", "cpi_yoy"], asof=asof).dropna()
    growth = (panel["gdp_qoq"] - panel["gdp_qoq"].mean()) / panel["gdp_qoq"].std()
    infl = (panel["cpi_yoy"] - panel["cpi_yoy"].mean()) / panel["cpi_yoy"].std()
    labels = []
    for g, p in zip(growth, infl):
        if g >= 0 and p < 0:
            labels.append("골디락스")
        elif g >= 0 and p >= 0:
            labels.append("리플레이션")
        elif g < 0 and p >= 0:
            labels.append("스태그플레이션")
        else:
            labels.append("디플레이션")
    return pd.Series(labels, index=panel.index, name="bad_regime")


def _vintage_scores(labeler, years: list[int]) -> pd.DataFrame:
    final = labeler(None)
    rows = []
    for year in years:
        cutoff = f"{year}-12-31"
        vintage = labeler(cutoff)
        if vintage.dropna().empty:
            continue
        compare_to = min(vintage.dropna().index[-1], pd.Timestamp(f"{year}-12-31"))
        common_final = final.loc[:compare_to]
        common_vintage = vintage.loc[:compare_to]
        if len(common_final.dropna()) < 24:
            continue
        rows.append(
            {
                "vintage_year": year,
                "n": int(len(common_final.dropna().index.intersection(common_vintage.dropna().index))),
                "agreement": _agreement(common_final, common_vintage),
                "kappa": _cohen_kappa(common_final, common_vintage),
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    cfg = LabelConfig(use_growth_composite=True, confirm_months=3, momentum_window=6)
    full_labels = make_regime_labels(cfg)
    stored_labels = load_regime_history()
    first_year = max(2010, int(full_labels.dropna().index[0].year) + 5)
    last_year = int(full_labels.dropna().index[-1].year) - 1
    years = list(range(first_year, last_year + 1))

    print("=" * 72)
    print("Regime Label Vintage Gate")
    print("=" * 72)
    print(
        "safe labeler config: "
        f"growth=composite, confirm={cfg.confirm_months}, momentum={cfg.momentum_window}"
    )

    stored_agree = _agreement(full_labels, stored_labels)
    print(f"[L0] stored labels vs regenerated safe labels agreement: {stored_agree:.3f}")
    if stored_agree < 0.95:
        print("     WARN: stored labels may use a different approved label config.")

    safe = _vintage_scores(lambda asof: make_regime_labels(cfg, asof=asof), years)
    assert not safe.empty
    min_agree = float(safe["agreement"].min())
    min_kappa = float(safe["kappa"].min())
    print(f"[L1] safe labeler min agreement={min_agree:.3f}, min kappa={min_kappa:.3f}")
    assert min_agree >= 0.95
    assert min_kappa >= 0.90
    print("[L1] safe labeler vintage stability: PASS")

    bad = _vintage_scores(_bad_full_sample_labeler, years)
    assert not bad.empty
    bad_min_agree = float(bad["agreement"].min())
    bad_min_kappa = float(bad["kappa"].min())
    print(
        f"[L2] negative control min agreement={bad_min_agree:.3f}, "
        f"min kappa={bad_min_kappa:.3f}"
    )
    assert bad_min_agree < 0.95 or bad_min_kappa < 0.90
    print("[L2] unsafe full-sample labeler is detected: PASS")

    out_dir = ROOT / "tests" / "analysis_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    safe.to_csv(out_dir / "label_vintage_safe_scores.csv", index=False)
    bad.to_csv(out_dir / "label_vintage_negative_control.csv", index=False)
    print(f"saved: {out_dir / 'label_vintage_safe_scores.csv'}")
    print("ALL PASS (실증 검증)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
