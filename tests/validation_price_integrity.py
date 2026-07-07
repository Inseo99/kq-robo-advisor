"""Price-data integrity validation.

This is the price-data counterpart to market-cap PiT, macro PiT, and regime-label
validation. It checks whether the price panel used by backtests/screeners is
adjusted enough to avoid split-like distortions.

Run:
    python tests\validation_price_integrity.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kq_tool.data.price_data import (  # noqa: E402
    adjust_for_actions,
    audit_price_panel,
    detect_corporate_actions,
    find_stale_series,
    krx_daily_limit,
)

FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}" + (f"  -- {detail}" if detail and not cond else ""))
    if not cond:
        FAILURES.append(f"{name}: {detail}")


def synthetic_prices(seed: int = 5) -> tuple[pd.DataFrame, pd.Timestamp]:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2014-01-02", "2026-06-05")
    n = len(idx)

    split_date = pd.Timestamp("2018-05-04")
    raw = 1_300_000 * np.exp(np.cumsum(rng.normal(0.0003, 0.018, n)))
    split_series = pd.Series(raw, index=idx)
    split_series.loc[idx >= split_date] = split_series.loc[idx >= split_date] / 50.0

    crash = pd.Series(50_000 * np.exp(np.cumsum(rng.normal(0.0002, 0.02, n))), index=idx)
    crash_date = pd.Timestamp("2020-03-19")
    crash_i = idx.get_loc(crash_date)
    desired_price = crash.iloc[crash_i - 1] * (1 - 0.29)
    crash.iloc[crash_i:] = crash.iloc[crash_i:] * desired_price / crash.iloc[crash_i]

    delisted = pd.Series(150_000 * np.exp(np.cumsum(rng.normal(0.0004, 0.02, n))), index=idx)
    delisted.loc[idx > pd.Timestamp("2023-08-14")] = np.nan

    return pd.DataFrame({"A_split50": split_series, "B_crash": crash, "C_delisted": delisted}), split_date


def test_v0_limits() -> None:
    check("V0a: 2015-06-15 이전 가격제한폭 15%", krx_daily_limit("2015-06-12") == 0.15)
    check("V0b: 2015-06-15 이후 가격제한폭 30%", krx_daily_limit("2015-06-15") == 0.30)


def test_v1_detector(prices: pd.DataFrame, split_date: pd.Timestamp) -> None:
    actions = detect_corporate_actions(prices["A_split50"])
    ok = len(actions) == 1 and actions[0].date == split_date and 45 <= actions[0].ratio <= 55 and actions[0].kind == "split_like"
    detail = [(a.date.date(), round(a.ratio, 1), a.kind) for a in actions]
    check("V1a: 50:1 split detected at exact date/ratio", ok, f"detected={detail}")

    crash_actions = detect_corporate_actions(prices["B_crash"])
    detail = [(a.date.date(), a.kind) for a in crash_actions]
    check("V1b: -29% crash within daily limit is not false-positive", len(crash_actions) == 0, f"detected={detail}")


def test_v2_adjust(prices: pd.DataFrame, split_date: pd.Timestamp) -> None:
    raw = prices["A_split50"].dropna()
    adjusted = adjust_for_actions(raw)

    check("V2a: no split-like jumps remain after adjustment", len(detect_corporate_actions(adjusted)) == 0)

    raw_returns = raw.pct_change().drop(index=split_date).dropna()
    adjusted_returns = adjusted.pct_change().drop(index=split_date).dropna()
    common = raw_returns.index.intersection(adjusted_returns.index)
    check("V2b: non-event daily returns are preserved", bool(np.allclose(raw_returns.loc[common], adjusted_returns.loc[common], atol=1e-12)))

    look_date = raw.index[raw.index.get_loc(split_date) + 5]
    prior_date = raw.index[raw.index.get_loc(look_date) - 252]
    raw_momentum = raw.loc[look_date] / raw.loc[prior_date] - 1
    adjusted_momentum = adjusted.loc[look_date] / adjusted.loc[prior_date] - 1
    check(
        "V2c: split-distorted 12M momentum is repaired",
        raw_momentum < -0.8 and adjusted_momentum > -0.5,
        f"raw={raw_momentum:.1%}, adjusted={adjusted_momentum:.1%}",
    )


def test_v3_stale(prices: pd.DataFrame) -> None:
    stale = find_stale_series(prices)
    check("V3: stale/delisted series is detected", set(stale) == {"C_delisted"}, f"detected={stale}")


def test_v4_real_panel() -> None:
    path = ROOT / "data" / "prices" / "close.csv"
    if not path.exists():
        print("[INFO] V4: data/prices/close.csv not found; real-panel audit skipped")
        print("       Expected wide format: first column date, remaining columns are tickers")
        return

    prices = pd.read_csv(path, parse_dates=["date"]).set_index("date")
    report = audit_price_panel(prices)
    out = ROOT / "tests" / "analysis_outputs" / "price_audit.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(out, index=False, encoding="utf-8-sig")

    n_unadjusted = int((report["issue"] == "unadjusted_action").sum())
    n_stale = int((report["issue"] == "stale_series").sum())
    print(f"[INFO] V4: audit report saved -> {out}")
    print(f"       unadjusted_action={n_unadjusted}, stale_series={n_stale}, total={len(report)}")
    check("V4: real panel has zero unadjusted corporate-action jumps", n_unadjusted == 0, f"{n_unadjusted} found")
    if n_stale:
        print("       stale tickers should be excluded after their last observation date in screeners/backtests")


def main() -> int:
    print("=" * 70)
    print("Price Data Integrity Validation")
    print("=" * 70)
    prices, split_date = synthetic_prices()
    test_v0_limits()
    test_v1_detector(prices, split_date)
    test_v2_adjust(prices, split_date)
    test_v3_stale(prices)
    test_v4_real_panel()
    print("-" * 70)
    if FAILURES:
        print(f"Result: FAIL ({len(FAILURES)} checks)")
        for failure in FAILURES:
            print("  -", failure)
        return 1
    print("Result: ALL PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
