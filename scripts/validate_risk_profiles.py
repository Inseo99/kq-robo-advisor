"""Validate suitability risk-profile presets before promotion.

This script is intentionally conservative. Until a real allocation backtest is
connected, non-neutral profiles remain DEFINED and are not promoted to live
recommendations. The output is still useful for presentation because it shows
the pre-registered promotion criteria and current status.

Usage:
  python scripts/validate_risk_profiles.py
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kq_tool.portfolio.risk_profiles import RISK_PROFILES, ProfileStatus  # noqa: E402


CRITERIA: dict[str, dict[str, object]] = {
    "stable": {
        "annual_vol_max": 0.05,
        "mdd_max": 0.10,
        "turnover_max": 1.5,
        "required_result": "Vol<=5%, MDD<=10%, turnover<=1.5x/year",
    },
    "neutral": {
        "status": "already_active_baseline",
        "required_result": "현행 추천 탭 기준. 기존 E2E/추천/ERC 게이트 유지",
    },
    "aggressive": {
        "sharpe_min": 0.50,
        "mdd_max": 0.35,
        "turnover_max": 3.0,
        "required_result": "Sharpe>=0.50, MDD<=35%, turnover<=3.0x/year",
    },
}


def run_allocation_backtest(profile_key: str) -> dict[str, float] | None:
    """Hook for the real ETF allocation backtest.

    Connect this function to the existing asset-allocation engine when the team
    is ready to validate stable/aggressive presets. Returning None keeps the
    profile in DEFINED status.
    """

    return None


def _passes(profile_key: str, metrics: Mapping[str, float] | None) -> tuple[bool, str]:
    if profile_key == "neutral":
        return True, "active baseline"
    if not metrics:
        return False, "not_run"
    crit = CRITERIA[profile_key]
    failures: list[str] = []
    if "annual_vol_max" in crit and metrics.get("annual_vol", 99.0) > float(crit["annual_vol_max"]):
        failures.append("annual_vol")
    if "mdd_max" in crit and abs(metrics.get("mdd", -99.0)) > float(crit["mdd_max"]):
        failures.append("mdd")
    if "sharpe_min" in crit and metrics.get("sharpe", -99.0) < float(crit["sharpe_min"]):
        failures.append("sharpe")
    if "turnover_max" in crit and metrics.get("turnover_yr", 99.0) > float(crit["turnover_max"]):
        failures.append("turnover")
    return (not failures), ("pass" if not failures else "fail:" + ",".join(failures))


def build_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for key in ("stable", "neutral", "aggressive"):
        profile = RISK_PROFILES[key]
        metrics = run_allocation_backtest(key)
        passed, reason = _passes(key, metrics)
        promotion = "eligible" if passed and profile.status != ProfileStatus.ACTIVE else "hold"
        if profile.status == ProfileStatus.ACTIVE:
            promotion = "already_active"
        rows.append(
            {
                "profile": key,
                "label": profile.label,
                "current_status": profile.status.value,
                "criteria": CRITERIA[key]["required_result"],
                "metrics_available": bool(metrics),
                "validation_result": reason,
                "promotion": promotion,
                "note": profile.validation_note,
            }
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        default=str(ROOT / "tests" / "analysis_outputs" / "risk_profile_validation.csv"),
        help="CSV output path",
    )
    args = parser.parse_args(argv)

    rows = build_rows()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("Risk profile validation status")
    print("-" * 72)
    for row in rows:
        print(
            f"{row['profile']:10s} {row['label']:6s} "
            f"status={row['current_status']:8s} result={row['validation_result']} "
            f"promotion={row['promotion']}"
        )
    print(f"\nSaved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
