"""
Signal Quality + Alpha Decay segmented validation.

This script compares raw technical-signal Alpha Decay with quality-filtered
events across equity/ETF, buy/sell, large/mid-small, and limit-move settings.
It is a validation tool only; production scoring should change only after
large OOS/placebo runs remain stable after costs.
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
from scipy.optimize import curve_fit

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

import data_loader as dl
import server
from kq_tool.analyzer.signal_quality import signal_quality_scores
from kq_tool.validation.costs import apply_signal_costs, net_edge_return, signal_trade_cost_rate
from kq_tool.validation.segments import segment_split_for_top, signal_names_for_side, ticker_segment


IS_START = pd.Timestamp("2014-01-01")
IS_END = pd.Timestamp("2020-12-31")
OOS_START = pd.Timestamp("2021-01-01")
HORIZONS = [1, 3, 5, 7, 10, 15, 20]
ETF_CACHE_PATH = os.path.join(ROOT, "data", "cache", "etf_assets_v2.parquet")
SIGNALS = {
    "RSI 과매도": (+1, "s_rsi"),
    "RSI 과매수": (-1, "s_rsi"),
    "MACD 골든크로스": (+1, "s_macd"),
    "MACD 데드크로스": (-1, "s_macd"),
    "BB 하단터치": (+1, "s_bb"),
    "BB 상단터치": (-1, "s_bb"),
}


def _exp(t, a, lam, c):
    return a * np.exp(-lam * t) + c


def parse_quality_grid(text: str) -> list[float]:
    values = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        value = float(part)
        if not 0.0 <= value <= 1.0:
            raise ValueError("quality threshold must be between 0 and 1")
        values.append(value)
    if not values:
        raise ValueError("at least one quality threshold is required")
    return values


def signal_masks(close: pd.Series) -> dict[str, pd.Series]:
    rsi = server._rsi(close)
    macd, sig = server._macd(close)
    bb_up, _, bb_low = server._bb(close)
    return {
        "RSI 과매도": rsi < 30,
        "RSI 과매수": rsi > 70,
        "MACD 골든크로스": (macd > sig) & (macd.shift(1) <= sig.shift(1)),
        "MACD 데드크로스": (macd < sig) & (macd.shift(1) >= sig.shift(1)),
        "BB 하단터치": close <= bb_low,
        "BB 상단터치": close >= bb_up,
    }


def get_equity_universe(top_n: int) -> tuple[list[str], str]:
    try:
        hist = server.get_mcap_history()
        tickers = server.get_top_mcap_at(hist, IS_END, n=top_n)
        if tickers:
            return tickers[:top_n], "IS_END 시점 시총 상위"
    except Exception:
        pass
    return server.get_top_marketcap_tickers(limit=top_n), "최근 시총 상위 fallback"


def get_etf_universe(top_n: int) -> tuple[list[str], str]:
    if not os.path.exists(ETF_CACHE_PATH):
        return [], "ETF cache missing"
    frame = pd.read_parquet(ETF_CACHE_PATH)
    return list(frame.columns[:top_n]), "ETF cache universe"


def load_equity_price_map(tickers: list[str]) -> dict[str, pd.DataFrame]:
    out = {}
    for index, ticker in enumerate(tickers, 1):
        code = server._ticker_to_code(ticker)
        frame = dl.get_ohlcv_df(code, adjusted=False)
        if frame is None or "Close" not in frame or frame["Close"].dropna().shape[0] < 260:
            continue
        frame = frame.copy()
        frame.index = pd.to_datetime(frame.index)
        out[ticker] = frame.sort_index()
        if index % 50 == 0:
            print(f"  가격 로드: {index}/{len(tickers)}")
    return out


def load_etf_price_map(tickers: list[str]) -> dict[str, pd.DataFrame]:
    if not tickers or not os.path.exists(ETF_CACHE_PATH):
        return {}
    close_frame = pd.read_parquet(ETF_CACHE_PATH)
    close_frame.index = pd.to_datetime(close_frame.index)
    close_frame = close_frame.sort_index()
    out = {}
    for ticker in tickers:
        if ticker not in close_frame:
            continue
        close = pd.to_numeric(close_frame[ticker], errors="coerce").dropna()
        if len(close) >= 260:
            out[ticker] = pd.DataFrame({"Close": close}, index=close.index)
    return out


def build_quality_cache(price_map: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    cache = {}
    for ticker, frame in price_map.items():
        close = frame["Close"].dropna()
        volume = frame["Volume"].reindex(close.index) if "Volume" in frame else None
        cache[ticker] = signal_quality_scores(close, volume=volume)
    return cache


def event_mask_for(
    frame: pd.DataFrame,
    signal_name: str,
    mode: str,
    min_quality: float,
    quality_scores: pd.DataFrame | None = None,
) -> pd.Series:
    close = frame["Close"].dropna()
    if mode == "raw":
        return signal_masks(close)[signal_name]
    if quality_scores is None:
        volume = frame["Volume"].reindex(close.index) if "Volume" in frame else None
        quality_scores = signal_quality_scores(close, volume=volume)
    return quality_scores[signal_name].reindex(close.index).fillna(0.0) >= min_quality


def valid_signal_dates(
    close: pd.Series,
    mask: pd.Series,
    start: pd.Timestamp,
    end: pd.Timestamp,
    limit_threshold: float,
) -> tuple[list[pd.Timestamp], int]:
    limit_hit = close.pct_change().abs() >= limit_threshold
    dates = []
    excluded = 0
    for dt in mask[mask].index:
        if dt < start or dt > end:
            continue
        idx = close.index.get_loc(dt)
        if isinstance(idx, slice):
            idx = idx.stop - 1
        bad_limit = bool(limit_hit.iloc[idx])
        if idx + 1 < len(close):
            bad_limit = bad_limit or bool(limit_hit.iloc[idx + 1])
        if bad_limit:
            excluded += 1
            continue
        dates.append(dt)
    return dates, excluded


def edge_return(close: pd.Series, dt: pd.Timestamp, horizon: int, direction: int) -> float | None:
    idx = close.index.get_loc(dt)
    if isinstance(idx, slice):
        idx = idx.stop - 1
    if idx + horizon >= len(close):
        return None
    p0 = float(close.iloc[idx])
    p1 = float(close.iloc[idx + horizon])
    if p0 <= 0 or not np.isfinite(p0) or not np.isfinite(p1):
        return None
    return direction * ((p1 - p0) / p0)


def estimate_decay_for_signals(
    price_map: dict[str, pd.DataFrame],
    mode: str,
    min_quality: float,
    limit_threshold: float,
    signal_names: list[str],
    quality_cache: dict[str, pd.DataFrame] | None = None,
) -> dict:
    stats = {}
    for signal_name in signal_names:
        direction, _ = SIGNALS[signal_name]
        hvals = {horizon: [] for horizon in HORIZONS}
        count = 0
        excluded_total = 0
        for ticker, frame in price_map.items():
            close = frame["Close"].dropna()
            scores = quality_cache.get(ticker) if quality_cache else None
            mask = event_mask_for(frame, signal_name, mode, min_quality, scores)
            dates, excluded = valid_signal_dates(close, mask, IS_START, IS_END, limit_threshold)
            excluded_total += excluded
            count += len(dates)
            for dt in dates:
                for horizon in HORIZONS:
                    ret = edge_return(close, dt, horizon, direction)
                    if ret is not None:
                        hvals[horizon].append(ret)

        avg = {h: (float(np.mean(values)) if values else None) for h, values in hvals.items()}
        valid = [(h, ret) for h, ret in avg.items() if ret is not None]
        half_life = None
        selected_day = None
        basis = "none"
        if len(valid) >= 4:
            try:
                x = np.array([h for h, _ in valid], dtype=float)
                y = np.array([ret for _, ret in valid], dtype=float)
                params, _ = curve_fit(_exp, x, y, p0=[y[0], 0.1, 0.0], maxfev=3000)
                if params[1] > 0:
                    raw_half_life = float(np.log(2) / params[1])
                    if 0 < raw_half_life <= max(HORIZONS) * 1.5:
                        half_life = round(raw_half_life, 1)
                        selected_day = int(max(1, min(max(HORIZONS), round(half_life))))
                        basis = "half_life"
            except Exception:
                pass
        if selected_day is None:
            positives = [(h, ret) for h, ret in valid if ret is not None and ret > 0]
            if positives:
                selected_day = int(max(positives, key=lambda item: item[1])[0])
                basis = "peak"
        stats[signal_name] = {
            "count": count,
            "excluded_limit": excluded_total,
            "half_life": half_life,
            "selected_day": selected_day,
            "basis": basis,
            "avg": {h: (round(avg[h] * 100, 3) if avg[h] is not None else None) for h in HORIZONS},
        }
    return stats


def collect_oos_events_for_signals(
    price_map: dict[str, pd.DataFrame],
    stats: dict,
    mode: str,
    min_quality: float,
    limit_threshold: float,
    signal_names: list[str],
    quality_cache: dict[str, pd.DataFrame] | None = None,
    cost_bps: float = 0.0,
    slippage_bps: float = 0.0,
    trade_sides: int = 2,
) -> dict[str, list[tuple[str, pd.Timestamp, int, int, float]]]:
    events = {signal_name: [] for signal_name in signal_names}
    for signal_name in signal_names:
        direction, _ = SIGNALS[signal_name]
        horizon = stats[signal_name].get("selected_day")
        if not horizon:
            continue
        for ticker, frame in price_map.items():
            close = frame["Close"].dropna()
            scores = quality_cache.get(ticker) if quality_cache else None
            mask = event_mask_for(frame, signal_name, mode, min_quality, scores)
            dates, _ = valid_signal_dates(close, mask, OOS_START, close.index[-1], limit_threshold)
            for dt in dates:
                ret = edge_return(close, dt, horizon, direction)
                if ret is not None:
                    net_ret = net_edge_return(
                        ret,
                        cost_bps=cost_bps,
                        slippage_bps=slippage_bps,
                        trade_sides=trade_sides,
                    )
                    if net_ret is not None:
                        events[signal_name].append((ticker, dt, horizon, direction, net_ret))
    return events


def random_return_pool(
    price_map: dict[str, pd.DataFrame],
    horizon: int,
    direction: int,
    limit_threshold: float,
    cost_bps: float = 0.0,
    slippage_bps: float = 0.0,
    trade_sides: int = 2,
) -> np.ndarray:
    values = []
    for frame in price_map.values():
        close = frame["Close"].dropna()
        limit_hit = close.pct_change().abs() >= limit_threshold
        idxs = np.flatnonzero(np.asarray(close.index >= OOS_START))
        idxs = idxs[idxs + horizon < len(close)]
        for idx in idxs:
            bad_limit = bool(limit_hit.iloc[idx])
            if idx + 1 < len(close):
                bad_limit = bad_limit or bool(limit_hit.iloc[idx + 1])
            if bad_limit:
                continue
            p0 = float(close.iloc[idx])
            p1 = float(close.iloc[idx + horizon])
            if p0 > 0 and np.isfinite(p0) and np.isfinite(p1):
                values.append(direction * ((p1 - p0) / p0))
    return np.array(
        apply_signal_costs(
            values,
            cost_bps=cost_bps,
            slippage_bps=slippage_bps,
            trade_sides=trade_sides,
        ),
        dtype=float,
    )


def summarize(
    events_by_signal: dict[str, list[tuple[str, pd.Timestamp, int, int, float]]],
    price_map: dict[str, pd.DataFrame],
    limit_threshold: float,
    n: int,
    seed: int,
    cost_bps: float = 0.0,
    slippage_bps: float = 0.0,
    trade_sides: int = 2,
) -> dict:
    rng = np.random.default_rng(seed)
    actual = []
    random_means = []
    signal_rows = []
    for signal_name, events in events_by_signal.items():
        if not events:
            signal_rows.append((signal_name, 0, None, None))
            continue
        returns = [float(event[4]) for event in events]
        actual.extend(returns)
        horizon = int(events[0][2])
        direction = int(events[0][3])
        pool = random_return_pool(
            price_map,
            horizon,
            direction,
            limit_threshold,
            cost_bps=cost_bps,
            slippage_bps=slippage_bps,
            trade_sides=trade_sides,
        )
        if len(pool) > 0:
            draws = [float(np.mean(rng.choice(pool, size=len(returns), replace=True))) for _ in range(n)]
            random_means.extend(draws)
            signal_rows.append((signal_name, len(returns), float(np.mean(returns)), float(np.mean(draws))))
        else:
            signal_rows.append((signal_name, len(returns), float(np.mean(returns)), None))
    actual_mean = float(np.mean(actual)) if actual else None
    random_mean = float(np.mean(random_means)) if random_means else None
    p_value = None
    if actual_mean is not None and random_means:
        p_value = float(np.mean(np.array(random_means) >= actual_mean))
    return {
        "actual_mean": actual_mean,
        "random_mean": random_mean,
        "p_value": p_value,
        "n_events": len(actual),
        "signal_rows": signal_rows,
    }


def run_mode_for_signals(
    price_map: dict[str, pd.DataFrame],
    mode: str,
    min_quality: float,
    limit_threshold: float,
    n: int,
    seed: int,
    signal_names: list[str],
    quality_cache: dict[str, pd.DataFrame] | None = None,
    cost_bps: float = 0.0,
    slippage_bps: float = 0.0,
    trade_sides: int = 2,
) -> dict:
    stats = estimate_decay_for_signals(price_map, mode, min_quality, limit_threshold, signal_names, quality_cache)
    events = collect_oos_events_for_signals(
        price_map,
        stats,
        mode,
        min_quality,
        limit_threshold,
        signal_names,
        quality_cache,
        cost_bps=cost_bps,
        slippage_bps=slippage_bps,
        trade_sides=trade_sides,
    )
    return {
        "stats": stats,
        "summary": summarize(
            events,
            price_map,
            limit_threshold,
            n,
            seed,
            cost_bps=cost_bps,
            slippage_bps=slippage_bps,
            trade_sides=trade_sides,
        ),
    }


def pct(value):
    return "n/a" if value is None else f"{value * 100:+.3f}%"


def save_summary_csv(
    out_path: str,
    raw_summary: dict,
    quality_results: list[dict],
    metadata: dict | None = None,
) -> str:
    csv_path = os.path.splitext(out_path)[0] + "_summary.csv"
    metadata = metadata or {}
    rows = [
        {
            "mode": "raw",
            "threshold": "",
            "events": raw_summary["n_events"],
            "edge_pct": None if raw_summary["actual_mean"] is None else raw_summary["actual_mean"] * 100,
            "random_pct": None if raw_summary["random_mean"] is None else raw_summary["random_mean"] * 100,
            "delta_vs_raw_pct": 0.0,
            "p_value": raw_summary["p_value"],
            "cost_bps": metadata.get("cost_bps", 0.0),
            "slippage_bps": metadata.get("slippage_bps", 0.0),
            "trade_sides": metadata.get("trade_sides", 2),
            "cost_rate_pct": metadata.get("cost_rate_pct", 0.0),
            **metadata,
        }
    ]
    raw_edge = raw_summary["actual_mean"]
    for row in quality_results:
        summary = row["summary"]
        edge = summary["actual_mean"]
        rows.append(
            {
                "mode": "quality",
                "threshold": row["threshold"],
                "events": summary["n_events"],
                "edge_pct": None if edge is None else edge * 100,
                "random_pct": None if summary["random_mean"] is None else summary["random_mean"] * 100,
                "delta_vs_raw_pct": None if raw_edge is None or edge is None else (edge - raw_edge) * 100,
                "p_value": summary["p_value"],
                "cost_bps": metadata.get("cost_bps", 0.0),
                "slippage_bps": metadata.get("slippage_bps", 0.0),
                "trade_sides": metadata.get("trade_sides", 2),
                "cost_rate_pct": metadata.get("cost_rate_pct", 0.0),
                **metadata,
            }
        )
    pd.DataFrame(rows).to_csv(csv_path, index=False, encoding="utf-8-sig")
    return csv_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top", type=int, default=50)
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--min-quality", type=float, default=0.6)
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--qualities", default="0.4,0.5,0.6,0.7,0.8")
    parser.add_argument("--limit-threshold", type=float, default=0.295)
    parser.add_argument("--include-limit-moves", action="store_true")
    parser.add_argument("--cost-bps", type=float, default=0.0)
    parser.add_argument("--slippage-bps", type=float, default=0.0)
    parser.add_argument("--trade-sides", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--universe-source", choices=["equity", "etf"], default="equity")
    parser.add_argument("--signal-side", choices=["all", "buy", "sell"], default="all")
    parser.add_argument("--ticker-segment", choices=["all", "large", "mid_small"], default="all")
    parser.add_argument("--segment-split", type=int, default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    thresholds = parse_quality_grid(args.qualities) if args.sweep else [float(args.min_quality)]
    signal_names = signal_names_for_side(SIGNALS, args.signal_side)
    split = segment_split_for_top(args.top, args.segment_split)
    limit_threshold = np.inf if args.include_limit_moves else args.limit_threshold
    cost_rate = signal_trade_cost_rate(
        cost_bps=args.cost_bps,
        slippage_bps=args.slippage_bps,
        trade_sides=args.trade_sides,
    )

    print("=" * 72)
    print("KQ Quant Tool — Signal Quality + Alpha Decay segmented validation")
    print("=" * 72)
    print(f"universe={args.universe_source}, top={args.top}, segment={args.ticker_segment}, split={split}")
    print(f"signal_side={args.signal_side}, signals={', '.join(signal_names)}")
    print("limit moves=" + ("included" if args.include_limit_moves else f"excluded threshold={args.limit_threshold}"))
    print(
        f"costs=cost {args.cost_bps}bps + slippage {args.slippage_bps}bps, "
        f"sides={args.trade_sides}, per-event haircut={cost_rate * 100:.3f}%"
    )

    started = time.time()
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
    print(f"  price available: {len(price_map)}")
    quality_cache = build_quality_cache(price_map)

    print("\n[2] RAW")
    raw_result = run_mode_for_signals(
        price_map,
        "raw",
        args.min_quality,
        limit_threshold,
        args.n,
        args.seed,
        signal_names,
        cost_bps=args.cost_bps,
        slippage_bps=args.slippage_bps,
        trade_sides=args.trade_sides,
    )
    raw = raw_result["summary"]
    print(f"  RAW events={raw['n_events']} edge={pct(raw['actual_mean'])} random={pct(raw['random_mean'])} p={raw['p_value']}")

    quality_results = []
    for threshold in thresholds:
        print(f"\n[3] QUALITY q>={threshold}")
        result = run_mode_for_signals(
            price_map,
            "quality",
            threshold,
            limit_threshold,
            args.n,
            args.seed,
            signal_names,
            quality_cache,
            cost_bps=args.cost_bps,
            slippage_bps=args.slippage_bps,
            trade_sides=args.trade_sides,
        )
        summary = result["summary"]
        print(
            f"  QUALITY events={summary['n_events']} edge={pct(summary['actual_mean'])} "
            f"random={pct(summary['random_mean'])} p={summary['p_value']}"
        )
        quality_results.append({"threshold": threshold, **result})

    print("\nSummary")
    print(f"RAW events={raw['n_events']} edge={pct(raw['actual_mean'])} random={pct(raw['random_mean'])}")
    for row in quality_results:
        summary = row["summary"]
        delta = None
        if raw["actual_mean"] is not None and summary["actual_mean"] is not None:
            delta = summary["actual_mean"] - raw["actual_mean"]
        print(
            f"QUALITY q>={row['threshold']:.2f} events={summary['n_events']} "
            f"edge={pct(summary['actual_mean'])} random={pct(summary['random_mean'])} delta={pct(delta)}"
        )

    out_path = args.output or os.path.join(os.path.dirname(__file__), "signal_quality_alpha_decay_results.npz")
    if not os.path.isabs(out_path):
        out_path = os.path.join(ROOT, out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(
        out_path,
        raw_summary=raw,
        quality_summaries=np.array(
            [{"threshold": row["threshold"], "summary": row["summary"], "stats": row["stats"]} for row in quality_results],
            dtype=object,
        ),
        universe_source=args.universe_source,
        signal_side=args.signal_side,
        ticker_segment=args.ticker_segment,
        include_limit_moves=bool(args.include_limit_moves),
        limit_threshold=limit_threshold,
        cost_bps=float(args.cost_bps),
        slippage_bps=float(args.slippage_bps),
        trade_sides=int(args.trade_sides),
        cost_rate=cost_rate,
        top=args.top,
        n=args.n,
        qualities=np.array(thresholds, dtype=float),
    )
    csv_path = save_summary_csv(
        out_path,
        raw,
        quality_results,
        metadata={
            "universe_source": args.universe_source,
            "signal_side": args.signal_side,
            "ticker_segment": args.ticker_segment,
            "segment_split": split,
            "include_limit_moves": bool(args.include_limit_moves),
            "limit_threshold": limit_threshold,
            "cost_bps": float(args.cost_bps),
            "slippage_bps": float(args.slippage_bps),
            "trade_sides": int(args.trade_sides),
            "cost_rate_pct": cost_rate * 100,
        },
    )
    print(f"\ncompleted in {time.time() - started:.1f}s")
    print(f"result: {out_path}")
    print(f"summary: {csv_path}")


if __name__ == "__main__":
    main()
