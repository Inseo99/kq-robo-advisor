r"""Fetch adjusted Korean close prices into data/prices/close.csv.

FinanceDataReader is preferred for Korean tickers because it usually provides
split-adjusted KRX prices. yfinance is available as a fallback and is forced to
use auto_adjust=True to avoid raw-close split distortions.

Examples:
    python scripts\fetch_prices.py --tickers 005930 000660 035420
    python scripts\fetch_prices.py --tickers-file data\price_tickers.txt
    python scripts\fetch_prices.py --source yfinance --tickers 005930.KS 000660.KS

After fetching, run:
    python tests\validation_price_integrity.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "prices" / "close.csv"


def normalize_ticker(ticker: str, source: str) -> str:
    text = str(ticker).strip()
    if not text:
        return text
    if source == "fdr":
        return text.split(".")[0].zfill(6) if text.split(".")[0].isdigit() else text
    if source == "yfinance" and text[:6].isdigit() and "." not in text:
        return f"{text[:6]}.KS"
    return text



def missing_ticker_file_message(path: Path) -> str:
    return (
        f"Ticker file not found: {path}\n"
        "Create it first, for example:\n"
        "  python -c \"import FinanceDataReader as fdr; "
        "df = fdr.StockListing('KOSPI'); "
        "df.nlargest(200, 'Marcap')['Code'].to_csv('tickers.txt', index=False, header=False)\"\n"
        "  Add-Content tickers.txt \"069500\",\"148070\",\"114260\",\"153130\",\"132030\",\"130680\",\"229200\"\n"
        "Then run:\n"
        "  python scripts\\fetch_prices.py --tickers-file tickers.txt --start 2014-01-01\n"
        "Or pass tickers directly:\n"
        "  python scripts\\fetch_prices.py --tickers 005930 000660 035420 --start 2014-01-01"
    )

def read_tickers(args: argparse.Namespace) -> list[str]:
    tickers: list[str] = []
    if args.tickers:
        tickers.extend(args.tickers)
    if args.tickers_file:
        path = Path(args.tickers_file)
        if not path.exists():
            raise FileNotFoundError(missing_ticker_file_message(path))
        for line in path.read_text(encoding="utf-8").splitlines():
            clean = line.strip()
            if clean and not clean.startswith("#"):
                tickers.append(clean.split(",")[0].strip())
    seen: set[str] = set()
    out: list[str] = []
    for ticker in tickers:
        normalized = normalize_ticker(ticker, args.source)
        if normalized and normalized not in seen:
            seen.add(normalized)
            out.append(normalized)
    return out


def fetch_fdr(ticker: str, start: str, end: str | None) -> pd.Series:
    try:
        import FinanceDataReader as fdr
    except ImportError as exc:
        raise RuntimeError("FinanceDataReader is not installed. Install it or use --source yfinance.") from exc

    frame = fdr.DataReader(ticker, start, end)
    if frame is None or frame.empty or "Close" not in frame:
        raise RuntimeError(f"No FDR Close data for {ticker}")
    close = pd.to_numeric(frame["Close"], errors="coerce").dropna()
    close.name = ticker
    return close


def fetch_yfinance(ticker: str, start: str, end: str | None) -> pd.Series:
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError("yfinance is not installed. Install it or use --source fdr.") from exc

    frame = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False, threads=False)
    if frame is None or frame.empty:
        raise RuntimeError(f"No yfinance data for {ticker}")
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(0)
    col = "Close" if "Close" in frame.columns else "Adj Close" if "Adj Close" in frame.columns else None
    if col is None:
        raise RuntimeError(f"No adjusted close column for {ticker}")
    close = pd.to_numeric(frame[col], errors="coerce").dropna()
    close.name = ticker
    return close


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", nargs="*", default=[])
    parser.add_argument("--tickers-file", default=None)
    parser.add_argument("--source", choices=["fdr", "yfinance"], default="fdr")
    parser.add_argument("--start", default="2010-01-01")
    parser.add_argument("--end", default=None)
    args = parser.parse_args()

    try:
        tickers = read_tickers(args)
    except FileNotFoundError as exc:
        print(f"[stop] {exc}")
        return 1

    if not tickers:
        print("[stop] No tickers provided. Use --tickers or --tickers-file.")
        print("       Example: python scripts\\fetch_prices.py --tickers 005930 000660 035420 --start 2014-01-01")
        return 1

    series: list[pd.Series] = []
    failures: list[tuple[str, str]] = []
    fetcher = fetch_fdr if args.source == "fdr" else fetch_yfinance
    for ticker in tickers:
        try:
            close = fetcher(ticker, args.start, args.end)
            series.append(close)
            print(f"[ok] {ticker}: {len(close)} rows ({close.index.min().date()} ~ {close.index.max().date()})")
        except Exception as exc:
            failures.append((ticker, str(exc)))
            print(f"[fail] {ticker}: {exc}")

    if not series:
        print("[stop] No price series downloaded.")
        return 1

    panel = pd.concat(series, axis=1).sort_index()
    panel.index.name = "date"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(OUT, encoding="utf-8-sig")
    print(f"\nsaved: {OUT}  shape={panel.shape}")
    if failures:
        print("\nFailures:")
        for ticker, message in failures:
            print(f"  - {ticker}: {message}")
        print("Delisted tickers may require historical KRX/manual supplementation to avoid survivorship bias.")
    print("\nnext: python tests\\validation_price_integrity.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

