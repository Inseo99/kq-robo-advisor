"""
export_dsr_inputs.py - DSR / regime-conditional analysis input exporter.

Outputs three shared CSV inputs under tests/:
  - strategy_monthly_returns.csv   rows: month-end, columns: strategy, values: monthly returns
  - asset_monthly_returns.csv      rows: month-end, columns: ETF/assets, values: monthly returns
  - regime_labels_monthly.csv      rows: month-end, column: regime, values: PiT regime label

Current project coverage:
  - 11 allocation/benchmark columns from validation_factor_regression
  - 3 strategy-backtest columns by default: quant, quant_s2, robo
  - 14 columns total unless more strategy-backtest keys are explicitly added

Run from the project root:
  python tests/export_dsr_inputs.py --start 2014-06-26

Notes:
  - equity/dates -> pct_change() follows validation_factor_regression.build_portfolio_returns().
  - regime labels use regime_pit.build_pit_regime_series(), a point-in-time classifier.
  - strategy keys are auto-discovered where possible. Use --extra-strategies quant,quant_s2,robo
    to force specific strategy-backtest keys.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TESTS_DIR = Path(__file__).resolve().parent
for p in (str(ROOT), str(TESTS_DIR), str(ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


DEFAULT_STRATEGY_KEYS = ["quant", "quant_s2", "robo"]


def export_allocation_returns(start_date: str | None, end_date: str | None) -> pd.DataFrame:
    """Export monthly returns for the existing 10 asset-allocation portfolios."""
    from validation_factor_regression import build_portfolio_returns  # noqa: E402

    df = build_portfolio_returns(start_date=start_date, end_date=end_date)
    df = df.sort_index()
    df.index = pd.to_datetime(df.index)
    return df


def export_asset_returns(start_date: str | None, end_date: str | None) -> pd.DataFrame:
    """Export monthly returns for the raw ETF/asset universe."""
    from validation_asset_allocation_v2 import load_etf_data  # noqa: E402

    etf = load_etf_data()
    etf.index = pd.to_datetime(etf.index)
    if start_date:
        etf = etf[etf.index >= pd.Timestamp(start_date)]
    if end_date:
        etf = etf[etf.index <= pd.Timestamp(end_date)]

    monthly = etf.resample("ME").last()
    rets = monthly.pct_change().replace([np.inf, -np.inf], np.nan)
    return rets.dropna(how="all")


def _equity_dict_to_monthly_returns(result: dict, name: str) -> pd.Series | None:
    """Convert server.run_strategy_backtest() output to a monthly return series."""
    if not isinstance(result, dict) or result.get("error"):
        msg = result.get("error") if isinstance(result, dict) else "no dict"
        print(f"    [skip] {name}: {msg}")
        return None

    equity = result.get("equity")
    dates = result.get("dates")
    if not equity or not dates or len(equity) != len(dates):
        print(f"    [skip] {name}: equity/dates mismatch")
        return None

    s = pd.Series(equity, index=pd.to_datetime(dates)).sort_index()
    s = s.resample("ME").last().dropna()
    rets = s.pct_change().dropna()
    return rets.astype(float) if not rets.empty else None


def _discover_strategy_keys(explicit: str | None) -> list[str]:
    """Choose strategy-backtest keys.

    If explicit comma-separated keys are provided, use them. Otherwise, try the
    strategy metadata registry and fall back to quant/quant_s2/robo.
    """
    if explicit:
        return [k.strip() for k in explicit.split(",") if k.strip()]

    keys = list(DEFAULT_STRATEGY_KEYS)
    try:
        from kq_tool.backtest import strategy_meta as sm  # noqa: E402

        for attr in ("STRATEGY_KEYS", "ALL_STRATEGIES", "BACKTEST_STRATEGIES"):
            registry = getattr(sm, attr, None)
            if not registry:
                continue

            found = list(registry.keys()) if isinstance(registry, dict) else list(registry)
            for key in found:
                key = str(key)
                if key and key not in keys and "compare" not in key:
                    keys.append(key)
            break
    except Exception as exc:
        print(f"  [info] strategy_meta auto-discovery failed; using defaults: {exc}")

    return keys


def export_strategy_backtest_returns(
    strategy_keys: list[str],
    period: str,
    rebalance: str,
    top_n: int,
    transaction_cost_bps: float,
    slippage_bps: float,
) -> pd.DataFrame:
    """Export monthly returns for server strategy-backtest strategies."""
    import server  # noqa: E402

    series_by_name = {}
    for key in strategy_keys:
        if "compare" in key:
            continue
        try:
            result = server.run_strategy_backtest(
                strategy=key,
                top_n=top_n,
                rebalance=rebalance,
                period=period,
                transaction_cost_bps=transaction_cost_bps,
                slippage_bps=slippage_bps,
            )
        except Exception as exc:
            print(f"    [skip] {key}: execution exception {exc}")
            continue

        rets = _equity_dict_to_monthly_returns(result, key)
        if rets is not None:
            series_by_name[key] = rets
            print(f"    [ok] {key}: {len(rets)} months")

    if not series_by_name:
        return pd.DataFrame()

    df = pd.DataFrame(series_by_name).sort_index()
    df.index = pd.to_datetime(df.index)
    return df


def export_regime_labels(month_end_index: pd.Index, min_quarters: int = 4) -> pd.DataFrame:
    """Export PiT monthly regime labels aligned to the strategy return index."""
    import data_loader as dl  # noqa: E402
    from regime_pit import build_pit_regime_series  # noqa: E402

    macro = dl.load_macro()
    series = build_pit_regime_series(macro, month_end_index, min_quarters=min_quarters)
    out = series.to_frame(name="regime")
    out.index.name = "date"
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2014-06-26")
    parser.add_argument("--end", default=None)
    parser.add_argument("--period", default="max", help="Strategy backtest period key, e.g. max/12y/3y.")
    parser.add_argument("--rebalance", default="M")
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--tc-bps", type=float, default=0.0)
    parser.add_argument("--slip-bps", type=float, default=0.0)
    parser.add_argument(
        "--extra-strategies",
        default=None,
        help="Comma-separated strategy keys to force, e.g. quant,quant_s2,robo.",
    )
    parser.add_argument("--min-quarters", type=int, default=4)
    parser.add_argument(
        "--expected-min-strategies",
        type=int,
        default=14,
        help="Warn only if the exported strategy count is below this project-specific floor.",
    )
    args = parser.parse_args()

    t0 = time.time()
    print("=" * 72)
    print("DSR / regime-conditional shared input export")
    print("=" * 72)

    print("\n[1] Asset-allocation monthly returns")
    allocation_returns = export_allocation_returns(args.start, args.end)
    print(f"    {allocation_returns.shape[1]} allocation strategies, {allocation_returns.shape[0]} months")

    print("\n[2] Strategy-backtest monthly returns")
    strategy_keys = _discover_strategy_keys(args.extra_strategies)
    print(f"    target strategy keys: {strategy_keys}")
    strategy_bt_returns = export_strategy_backtest_returns(
        strategy_keys,
        args.period,
        args.rebalance,
        args.top_n,
        args.tc_bps,
        args.slip_bps,
    )
    print(f"    {strategy_bt_returns.shape[1]} strategy-backtest columns, {strategy_bt_returns.shape[0]} months")

    strategy_all = allocation_returns.join(strategy_bt_returns, how="outer").sort_index()
    strategy_all = strategy_all.loc[:, ~strategy_all.columns.duplicated()]
    print(f"\n    -> unified matrix: {strategy_all.shape[1]} strategies x {strategy_all.shape[0]} months")
    print(f"       strategies: {list(strategy_all.columns)}")

    print("\n[3] Raw ETF/asset monthly returns")
    asset_returns = export_asset_returns(args.start, args.end)
    print(f"    {asset_returns.shape[1]} assets, {asset_returns.shape[0]} months")

    print("\n[4] PiT regime labels")
    regimes = export_regime_labels(strategy_all.index, min_quarters=args.min_quarters)
    n_missing = int(regimes["regime"].isna().sum())
    print(f"    {len(regimes)} monthly labels (missing from data limits: {n_missing})")
    counts = regimes["regime"].value_counts()
    print("    regime distribution:")
    print(counts.to_string() if not counts.empty else "    <empty>")

    strategy_path = TESTS_DIR / "strategy_monthly_returns.csv"
    asset_path = TESTS_DIR / "asset_monthly_returns.csv"
    regime_path = TESTS_DIR / "regime_labels_monthly.csv"
    strategy_all.to_csv(strategy_path, encoding="utf-8-sig")
    asset_returns.to_csv(asset_path, encoding="utf-8-sig")
    regimes.to_csv(regime_path, encoding="utf-8-sig")

    print("\n" + "-" * 72)
    print(f"Saved: {strategy_path}  ({strategy_all.shape[1]} strategies)")
    print(f"Saved: {asset_path}  ({asset_returns.shape[1]} assets)")
    print(f"Saved: {regime_path}")
    print(f"Elapsed: {time.time() - t0:.1f}s")

    if strategy_all.shape[1] < args.expected_min_strategies:
        print(
            f"\n[!] Only {strategy_all.shape[1]} strategies were exported; "
            f"expected at least {args.expected_min_strategies}."
        )
        print("    Pass additional strategy-backtest keys with --extra-strategies if available.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
