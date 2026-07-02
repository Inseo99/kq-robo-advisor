"""
Regime-segmented Alpha Decay validation.

This validation answers a narrow question:
Do Alpha Decay technical signals keep their OOS edge inside each point-in-time
macro regime, or is the edge mostly a market-regime coincidence?
"""

from __future__ import annotations

import argparse
import os
import sys
import time

os.environ["KQ_DISABLE_TABPFN"] = "1"
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

import data_loader as dl
from kq_tool.validation.costs import signal_trade_cost_rate
from kq_tool.validation.regime_alpha_decay import regime_summary_rows, summarize_regime_alpha_decay
from kq_tool.validation.segments import segment_split_for_top, signal_names_for_side, ticker_segment
from regime_pit import REGIMES, build_pit_regime_series
from validation_signal_quality_alpha_decay import (
    OOS_START,
    SIGNALS,
    build_quality_cache,
    collect_oos_events_for_signals,
    estimate_decay_for_signals,
    get_equity_universe,
    get_etf_universe,
    load_equity_price_map,
    load_etf_price_map,
    pct,
)


def _all_oos_dates(price_map: dict[str, pd.DataFrame]) -> pd.DatetimeIndex:
    dates: set[pd.Timestamp] = set()
    for frame in price_map.values():
        if "Close" not in frame:
            continue
        close = frame["Close"].dropna()
        close.index = pd.to_datetime(close.index)
        dates.update(pd.Timestamp(dt) for dt in close.index if pd.Timestamp(dt) >= OOS_START)
    return pd.DatetimeIndex(sorted(dates))


def _print_regime_table(rows: list[dict]) -> None:
    print("\n국면별 OOS Alpha Decay 결과")
    print("-" * 88)
    print(f"{'국면':<16} {'events':>8} {'edge':>12} {'random':>12} {'excess':>12} {'pctile':>10} {'p':>10}")
    for row in rows:
        edge = "n/a" if row["edge_pct"] is None else f"{row['edge_pct']:+.3f}%"
        random = "n/a" if row["random_pct"] is None else f"{row['random_pct']:+.3f}%"
        excess = "n/a" if row["excess_pct"] is None else f"{row['excess_pct']:+.3f}%"
        percentile = "n/a" if row["percentile"] is None else f"{row['percentile']:.1f}%"
        p_value = "n/a" if row["p_value"] is None else f"{row['p_value']:.3f}"
        print(f"{row['regime']:<16} {row['events']:>8} {edge:>12} {random:>12} {excess:>12} {percentile:>10} {p_value:>10}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top", type=int, default=50)
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--mode", choices=["raw", "quality"], default="quality")
    parser.add_argument("--min-quality", type=float, default=0.6)
    parser.add_argument("--limit-threshold", type=float, default=0.295)
    parser.add_argument("--include-limit-moves", action="store_true")
    parser.add_argument("--cost-bps", type=float, default=0.0)
    parser.add_argument("--slippage-bps", type=float, default=0.0)
    parser.add_argument("--trade-sides", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--universe-source", choices=["equity", "etf"], default="equity")
    parser.add_argument("--signal-side", choices=["all", "buy", "sell"], default="buy")
    parser.add_argument("--ticker-segment", choices=["all", "large", "mid_small"], default="all")
    parser.add_argument("--segment-split", type=int, default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    started = time.time()
    split = segment_split_for_top(args.top, args.segment_split)
    signal_names = signal_names_for_side(SIGNALS, args.signal_side)
    limit_threshold = np.inf if args.include_limit_moves else args.limit_threshold
    cost_rate = signal_trade_cost_rate(
        cost_bps=args.cost_bps,
        slippage_bps=args.slippage_bps,
        trade_sides=args.trade_sides,
    )

    print("=" * 72)
    print("KQ Quant Tool — Regime Alpha Decay validation")
    print("=" * 72)
    print(f"universe={args.universe_source}, top={args.top}, segment={args.ticker_segment}, split={split}")
    print(f"mode={args.mode}, min_quality={args.min_quality}, signal_side={args.signal_side}")
    print(f"signals={', '.join(signal_names)}")
    print("limit moves=" + ("included" if args.include_limit_moves else f"excluded threshold={args.limit_threshold}"))
    print(
        f"costs=cost {args.cost_bps}bps + slippage {args.slippage_bps}bps, "
        f"sides={args.trade_sides}, per-event haircut={cost_rate * 100:.3f}%"
    )

    if args.universe_source == "etf":
        tickers, universe_note = get_etf_universe(args.top)
        loader = load_etf_price_map
    else:
        tickers, universe_note = get_equity_universe(args.top)
        loader = load_equity_price_map
    tickers = ticker_segment(tickers, args.ticker_segment, split)
    if not tickers:
        raise SystemExit("selected ticker segment is empty")

    print(f"\n[1] {universe_note}: {len(tickers)} tickers")
    price_map = loader(tickers)
    if not price_map:
        raise SystemExit("no price data available for selected universe")
    print(f"  price available: {len(price_map)}")

    quality_cache = build_quality_cache(price_map) if args.mode == "quality" else None

    print("\n[2] Alpha Decay IS half-life/selected horizon 추정")
    stats = estimate_decay_for_signals(
        price_map,
        args.mode,
        args.min_quality,
        limit_threshold,
        signal_names,
        quality_cache,
    )
    for name in signal_names:
        row = stats.get(name, {})
        print(
            f"  {name}: IS count={row.get('count', 0)}, "
            f"selected_day={row.get('selected_day')}, basis={row.get('basis')}"
        )

    print("\n[3] OOS 이벤트 수집")
    events = collect_oos_events_for_signals(
        price_map,
        stats,
        args.mode,
        args.min_quality,
        limit_threshold,
        signal_names,
        quality_cache,
        cost_bps=args.cost_bps,
        slippage_bps=args.slippage_bps,
        trade_sides=args.trade_sides,
    )
    total_events = sum(len(items) for items in events.values())
    print(f"  OOS events={total_events}")

    print("\n[4] PiT 매크로 국면 생성")
    oos_dates = _all_oos_dates(price_map)
    if oos_dates.empty:
        raise SystemExit("no OOS dates available")
    regime_series = build_pit_regime_series(dl.load_macro(), oos_dates)
    print("  국면 분포:")
    for regime, count in regime_series.value_counts(dropna=False).items():
        print(f"    {regime}: {count}")

    print("\n[5] 같은 국면 내 무작위 날짜 placebo 비교")
    summary = summarize_regime_alpha_decay(
        events,
        price_map,
        regime_series,
        limit_threshold=limit_threshold,
        n=args.n,
        seed=args.seed,
        oos_start=OOS_START,
        regimes=REGIMES,
        cost_bps=args.cost_bps,
        slippage_bps=args.slippage_bps,
        trade_sides=args.trade_sides,
    )
    rows = regime_summary_rows(summary)
    _print_regime_table(rows)

    out_path = args.output or os.path.join(ROOT, "data", "validation", "regime_alpha_decay.npz")
    if not os.path.isabs(out_path):
        out_path = os.path.join(ROOT, out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(
        out_path,
        summary=summary,
        rows=np.array(rows, dtype=object),
        stats=stats,
        events=events,
        regime_counts=regime_series.value_counts(dropna=False).to_dict(),
        universe_source=args.universe_source,
        signal_side=args.signal_side,
        ticker_segment=args.ticker_segment,
        mode=args.mode,
        min_quality=float(args.min_quality),
        include_limit_moves=bool(args.include_limit_moves),
        limit_threshold=limit_threshold,
        cost_bps=float(args.cost_bps),
        slippage_bps=float(args.slippage_bps),
        trade_sides=int(args.trade_sides),
        cost_rate=cost_rate,
        top=args.top,
        n=args.n,
    )
    csv_path = os.path.splitext(out_path)[0] + "_summary.csv"
    pd.DataFrame(rows).assign(
        universe_source=args.universe_source,
        signal_side=args.signal_side,
        ticker_segment=args.ticker_segment,
        mode=args.mode,
        min_quality=float(args.min_quality),
        cost_bps=float(args.cost_bps),
        slippage_bps=float(args.slippage_bps),
        trade_sides=int(args.trade_sides),
        cost_rate_pct=cost_rate * 100,
        top=int(args.top),
        n=int(args.n),
    ).to_csv(csv_path, index=False, encoding="utf-8-sig")

    print("\n해석 기준:")
    print("  edge > random 이고 p가 낮으면, 해당 국면에서 Alpha Decay 신호가 무작위 날짜보다 낫다는 뜻입니다.")
    print("  events가 적은 국면은 결론을 보류하고 표본을 늘려야 합니다.")
    print(f"\ncompleted in {time.time() - started:.1f}s")
    print(f"result: {out_path}")
    print(f"summary: {csv_path}")


if __name__ == "__main__":
    main()


