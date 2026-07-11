"""Daily yfinance fundamental cache for the Kang-style screener.

The app server must not call live yfinance from the screener path.  Run this
script separately, then the server reads only data/cache/fundamentals_yf.parquet.
Failed tickers are retained as rows with missing fields and an error message so
coverage can be audited in the UI.
"""

from __future__ import annotations

import argparse
import datetime as dt
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_OUTPUT = ROOT / "data" / "cache" / "fundamentals_yf.parquet"


def _as_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _first_mapping_value(mapping: dict[str, Any] | None, keys: tuple[str, ...]) -> float | None:
    if not isinstance(mapping, dict):
        return None
    for key in keys:
        value = _as_float(mapping.get(key))
        if value is not None:
            return value
    return None


def _first_statement_value(frame: Any, labels: tuple[str, ...]) -> float | None:
    if frame is None or getattr(frame, "empty", True):
        return None
    try:
        index_map = {str(idx).strip().lower(): idx for idx in frame.index}
        for label in labels:
            idx = index_map.get(label.lower())
            if idx is None:
                continue
            row = frame.loc[idx].dropna()
            if len(row):
                value = _as_float(row.iloc[0])
                if value is not None:
                    return value
    except Exception:
        return None
    return None


def _load_universe() -> dict[str, tuple[str, str]]:
    try:
        import server

        return dict(server.UNIVERSE)
    except Exception:
        from kq_tool.data.universe import FALLBACK_UNIVERSE

        return dict(FALLBACK_UNIVERSE)


def _fetch_one(ticker: str, name: str, sector: str) -> dict[str, Any]:
    import yfinance as yf

    row: dict[str, Any] = {
        "ticker": ticker,
        "code": ticker.split(".")[0],
        "name": name,
        "sector": sector,
        "fetched_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "operating_cashflow": None,
        "net_income_common": None,
        "total_assets": None,
        "operating_income": None,
        "market_cap": None,
        "shares_outstanding": None,
        "current_price": None,
        "error": None,
    }
    try:
        obj = yf.Ticker(ticker)
        info = obj.info or {}
        cashflow = getattr(obj, "cashflow", None)
        income_stmt = getattr(obj, "income_stmt", None)
        balance_sheet = getattr(obj, "balance_sheet", None)

        row["operating_cashflow"] = (
            _first_mapping_value(info, ("operatingCashflow", "totalCashFromOperatingActivities"))
            or _first_statement_value(cashflow, ("Operating Cash Flow", "Total Cash From Operating Activities"))
        )
        row["net_income_common"] = (
            _first_mapping_value(info, ("netIncomeToCommon", "netIncome"))
            or _first_statement_value(income_stmt, ("Net Income Common Stockholders", "Net Income"))
        )
        row["total_assets"] = (
            _first_mapping_value(info, ("totalAssets",))
            or _first_statement_value(balance_sheet, ("Total Assets",))
        )
        row["operating_income"] = (
            _first_mapping_value(info, ("operatingIncome",))
            or _first_statement_value(income_stmt, ("Operating Income",))
        )
        row["market_cap"] = _first_mapping_value(info, ("marketCap",))
        row["shares_outstanding"] = _first_mapping_value(
            info,
            ("sharesOutstanding", "impliedSharesOutstanding"),
        )
        row["current_price"] = _first_mapping_value(
            info,
            ("currentPrice", "regularMarketPrice", "previousClose"),
        )
    except Exception as exc:
        row["error"] = str(exc)
    return row


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build data/cache/fundamentals_yf.parquet")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--limit", type=int, default=None, help="debug용 상위 N개만 수집")
    parser.add_argument("--sleep", type=float, default=0.05, help="ticker 사이 대기 초")
    args = parser.parse_args(argv)

    universe = _load_universe()
    items = list(universe.items())
    if args.limit:
        items = items[: args.limit]

    rows = []
    for i, (ticker, meta) in enumerate(items, start=1):
        name = meta[0] if len(meta) > 0 else ticker
        sector = meta[1] if len(meta) > 1 else ""
        row = _fetch_one(ticker, name, sector)
        rows.append(row)
        ok = all(_as_float(row.get(k)) is not None for k in (
            "operating_cashflow",
            "net_income_common",
            "total_assets",
            "operating_income",
        ))
        print(f"[{i:>4}/{len(items)}] {ticker} {'OK' if ok else 'MISS'} {row.get('error') or ''}")
        if args.sleep:
            time.sleep(args.sleep)

    df = pd.DataFrame(rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output, index=False)
    strict = df[["operating_cashflow", "net_income_common", "total_assets", "operating_income"]].notna().all(axis=1)
    print(f"saved: {output}")
    print(f"coverage: {int(strict.sum())}/{len(df)} ({strict.mean():.1%})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
