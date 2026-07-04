"""App wiring validation.

This guards the UI/server data plumbing so the app reads the same adjusted
prices and macro files as the validation pipeline.

Run:
  python tests\validation_app_wiring.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kq_tool.data.app_data import (  # noqa: E402
    get_macro_snapshot,
    get_price_frame,
    get_price_series,
    get_regime_payload,
    get_universe,
    load_close_panel,
)

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}" + (f" -- {detail}" if detail and not condition else ""))
    if not condition:
        FAILURES.append(name)


def main() -> int:
    print("=" * 72)
    print("App wiring validation")
    print("=" * 72)

    panel = load_close_panel()
    check("W1a: adjusted close panel loads", not panel.empty and "005930" in panel.columns)
    samsung = get_price_series("005930.KS", "max")
    max_abs_daily = float(samsung.pct_change().abs().dropna().max()) if samsung is not None else 999.0
    check("W1b: Samsung adjusted series has no split cliff", max_abs_daily < 0.40, f"max daily move={max_abs_daily:.3f}")
    frame = get_price_frame("005930.KS", "1y")
    check("W1c: app price frame is OHLCV-like and non-negative", frame is not None and set(["Open", "High", "Low", "Close", "Volume"]).issubset(frame.columns) and float(frame[["Open", "High", "Low", "Close"]].min().min()) > 0)

    active = set(get_universe())
    check("W2a: active universe comes from adjusted panel", "005930" in active and "069500" in active)
    check("W2b: stale/dead example Osstem is not active", "048260" not in active)

    macro = get_macro_snapshot()
    features = macro.get("current_features", {})
    values = [features.get("gdp_growth"), features.get("spread"), features.get("usd_change")]
    check("W3a: macro card values are loaded, not default zeros", all(v is not None for v in values) and any(abs(float(v)) > 1e-12 for v in values))
    check("W3b: GDP value is plausible decimal for UI percent formatting", features.get("gdp_growth") is not None and abs(float(features["gdp_growth"])) < 0.10)

    payload = get_regime_payload()
    check("W4a: regime payload uses single-source validity", payload.get("expected_remaining_days") == 205 and payload.get("validity_display") == "약 205일")
    check("W4b: shadow ledger is included when state exists", "shadow_ledger" in payload and payload["shadow_ledger"].get("band_vs_quarterly", {}).get("verdict") == "밴드 유지")

    server_text = (ROOT / "server.py").read_text(encoding="utf-8")
    index_text = (ROOT / "index.html").read_text(encoding="utf-8")
    check("W5a: server imports app_data gateway", "kq_tool.data.app_data" in server_text)
    check("W5b: chart path calls get_price_frame before legacy Excel", "_kq_app_get_price_frame" in server_text)
    check("W5c: legacy macro wording removed from UI", "HMM 12년 통계" not in index_text and "301분기" not in index_text)

    print("-" * 72)
    if FAILURES:
        print(f"Result: FAIL ({len(FAILURES)} checks)")
        return 1
    print("Result: ALL PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
