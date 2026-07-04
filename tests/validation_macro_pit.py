"""Macro point-in-time validation.

Checks that macro regime inputs respect publication lags:
  T1. docs/macro_publication_lags.md and INDICATOR_REGISTRY agree.
  T2. monthly/quarterly publication-type indicators have lag >= 1.
  T3. observable[t] == raw[t - lag] on synthetic data.
  T4. asof panels match a vintage simulation.
  T5. src code does not directly access _load_raw_* helpers.

Run:
  python tests/validation_macro_pit.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from kq_tool.regime.macro_data import (  # noqa: E402
    INDICATOR_REGISTRY,
    clear_raw_sources,
    get_observable_panel,
    register_raw_source,
    to_observable,
)

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}" + (f" -- {detail}" if detail and not condition else ""))
    if not condition:
        FAILURES.append(f"{name}: {detail}")


def test_registry_matches_doc() -> None:
    doc_path = ROOT / "docs" / "macro_publication_lags.md"
    text = doc_path.read_text(encoding="utf-8")

    doc_lags: dict[str, int] = {}
    for line in text.splitlines():
        match = re.match(r"^\|\s*`(\w+)`\s*\|", line)
        if not match:
            continue
        key = match.group(1)
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        applied = cells[-1]
        lag_match = re.search(r"(\d+)", applied)
        if lag_match:
            doc_lags[key] = int(lag_match.group(1))

    check("T1a: document indicator rows parsed", len(doc_lags) > 0, "check markdown table format")
    reg_keys = set(INDICATOR_REGISTRY)
    doc_keys = set(doc_lags)
    check("T1b: every registry indicator is documented", reg_keys <= doc_keys, f"missing: {sorted(reg_keys - doc_keys)}")
    check("T1c: every documented indicator is registered", doc_keys <= reg_keys, f"missing: {sorted(doc_keys - reg_keys)}")

    mismatch = {
        key: (doc_lags[key], INDICATOR_REGISTRY[key].total_lag_m)
        for key in reg_keys & doc_keys
        if doc_lags[key] != INDICATOR_REGISTRY[key].total_lag_m
    }
    check("T1d: documented lag equals code total_lag", not mismatch, f"mismatch: {mismatch}")


def test_minimum_lag() -> None:
    bad = [
        key
        for key, spec in INDICATOR_REGISTRY.items()
        if spec.freq in ("M", "Q") and spec.total_lag_m < 1
    ]
    check("T2: monthly/quarterly published indicators have lag >= 1", not bad, f"lag=0 published indicators: {bad}")


def _synthetic_monthly(n: int = 60, seed: int = 0) -> pd.Series:
    idx = pd.date_range("2020-01-31", periods=n, freq="ME")
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(size=n).cumsum(), index=idx)


def _stable_seed(text: str) -> int:
    return sum((idx + 1) * ord(ch) for idx, ch in enumerate(text)) % 2**32


def test_shift_correctness() -> None:
    for key, spec in INDICATOR_REGISTRY.items():
        raw = _synthetic_monthly(seed=_stable_seed(key))
        if spec.freq == "Q":
            raw = raw[raw.index.month.isin([3, 6, 9, 12])].asfreq("ME")

        obs = to_observable(raw, key)
        lag = spec.total_lag_m
        ok = True
        for date in obs.index[lag:]:
            expected_date = date - pd.offsets.MonthEnd(lag)
            expected = raw.get(expected_date, np.nan)
            actual = obs.loc[date]
            if spec.freq == "Q":
                past = raw.loc[:expected_date].dropna()
                expected = past.iloc[-1] if len(past) else np.nan
            if pd.isna(expected) and pd.isna(actual):
                continue
            if pd.isna(expected) != pd.isna(actual) or not np.isclose(expected, actual):
                ok = False
                break
        check(f"T3: shift correctness [{key}] lag={lag} freq={spec.freq}", ok, "observable[t] != raw[t-lag]")

        if lag > 0 and spec.freq != "Q":
            leak = any(
                not pd.isna(obs.get(date, np.nan))
                and np.isclose(obs.get(date, np.nan), raw.loc[date])
                and not np.isclose(raw.loc[date], raw.shift(lag).loc[date])
                for date in raw.index[lag:]
                if not pd.isna(raw.loc[date])
            )
            check(f"T3x: no same-month leakage [{key}]", not leak, "obs[t] contains raw[t]")


def test_vintage_simulation() -> None:
    clear_raw_sources()
    keys = list(INDICATOR_REGISTRY)
    full_raw: dict[str, pd.Series] = {}
    for key in keys:
        series = _synthetic_monthly(n=72, seed=_stable_seed("v" + key))
        if INDICATOR_REGISTRY[key].freq == "Q":
            series = series[series.index.month.isin([3, 6, 9, 12])]
        full_raw[key] = series

    def make_loader(series: pd.Series):
        return lambda: series

    for key in keys:
        register_raw_source(key, make_loader(full_raw[key]))

    full_panel = get_observable_panel(keys)
    ok_all = True
    detail = ""
    for asof in ("2021-06-30", "2022-12-31", "2024-03-31", "2025-01-15"):
        asof_ts = pd.Timestamp(asof)
        cutoff = asof_ts.to_period("M").to_timestamp("M")
        if asof_ts < cutoff:
            cutoff = cutoff - pd.offsets.MonthEnd(1)

        panel_a = get_observable_panel(keys, asof=asof)
        for key in keys:
            register_raw_source(key, make_loader(full_raw[key].loc[:cutoff]))
        panel_b = get_observable_panel(keys, asof=asof)
        for key in keys:
            register_raw_source(key, make_loader(full_raw[key]))

        same_a = panel_a.equals(full_panel.loc[:cutoff])
        same_b = panel_b.reindex_like(panel_a).equals(panel_a)
        if not (same_a and same_b):
            ok_all = False
            detail = f"asof={asof}: full-prefix={same_a}, vintage={same_b}"
            break

    clear_raw_sources()
    check("T4: vintage simulation no-lookahead", ok_all, detail)


FORBIDDEN = re.compile(r"_load_raw_(series|frame)")


def test_static_scan() -> None:
    offenders = []
    for py_file in (ROOT / "src").rglob("*.py"):
        if py_file.name == "macro_data.py":
            continue
        if FORBIDDEN.search(py_file.read_text(encoding="utf-8", errors="ignore")):
            offenders.append(str(py_file.relative_to(ROOT)))
    check("T5: src code does not use _load_raw_* directly", not offenders, f"offenders: {offenders}")


def main() -> int:
    print("=" * 70)
    print("Macro PiT validation")
    print("=" * 70)
    test_registry_matches_doc()
    test_minimum_lag()
    test_shift_correctness()
    test_vintage_simulation()
    test_static_scan()
    print("-" * 70)
    if FAILURES:
        print(f"결과: FAIL ({len(FAILURES)}건)")
        for failure in FAILURES:
            print("  -", failure)
        return 1
    print("결과: ALL PASS - publication-lag look-ahead guard is active")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
