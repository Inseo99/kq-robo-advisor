"""Fetch ECOS Open API series into data/macro/*.csv.

Preparation
-----------
1. Get a free ECOS API key from https://ecos.bok.or.kr/api/
2. Prefer setting it as an environment variable:
       $env:ECOS_API_KEY="YOUR_KEY"
   or pass it explicitly:
       python scripts\fetch_ecos.py --api-key YOUR_KEY --discover 817Y002
3. Use --discover to confirm item codes, then fill SERIES below.
4. Run:
       python scripts\fetch_ecos.py

Outputs use the common project format: data/macro/<key>.csv with columns
``date,value``. Derived series are generated after raw downloads:
``cpi_yoy``, ``yield_spread_10y_3y``, and ``credit_spread``.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Iterable

API_KEY = ""  # Optional local fallback. Prefer ECOS_API_KEY or --api-key.
BASE = "https://ecos.bok.or.kr/api"
ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "macro"

# key: (stat_code, cycle, item_code, start)
# Fill item_code=None values after checking with --discover.
SERIES: dict[str, tuple[str | None, str, str | None, str]] = {
    "base_rate": ("722Y001", "M", "0101000", "199901"),
    "cpi_index": ("901Y009", "M", "0", "199901"),
    "treasury_3y": ("817Y002", "D", None, "20000101"),
    "treasury_10y": ("817Y002", "D", None, "20000101"),
    "corp_aa_3y": ("817Y002", "D", None, "20000101"),
    "usdkrw": ("731Y001", "D", None, "20000101"),
    "kospi": ("802Y001", "D", None, "20000101"),
    "gdp_qoq": (None, "Q", None, "1999Q1"),
}


def resolve_api_key(cli_key: str | None = None) -> str:
    key = cli_key or os.getenv("ECOS_API_KEY") or API_KEY
    if not key or key.startswith("여기에"):
        raise RuntimeError(
            "ECOS API key is missing. Set ECOS_API_KEY, pass --api-key, "
            "or fill API_KEY in scripts/fetch_ecos.py."
        )
    return key


def _get_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def _ecos_url(path_parts: Iterable[str]) -> str:
    encoded = [urllib.parse.quote(str(part), safe="") for part in path_parts]
    return "/".join([BASE, *encoded])


def discover(stat_code: str, api_key: str) -> None:
    """Print ECOS StatisticItemList rows for one statistic code."""

    url = _ecos_url(["StatisticItemList", api_key, "json", "kr", 1, 1000, stat_code])
    data = _get_json(url)
    rows = data.get("StatisticItemList", {}).get("row", [])
    if not rows:
        print(f"No rows or API error: {json.dumps(data, ensure_ascii=False)[:500]}")
        return

    print(f"Statistic {stat_code} item list:")
    for row in rows:
        item_code = row.get("ITEM_CODE") or row.get("ITEM_CODE1") or ""
        item_name = row.get("ITEM_NAME") or row.get("ITEM_NAME1") or ""
        cycle = row.get("CYCLE", "?")
        print(f"  item_code={item_code:<16} cycle={cycle:<3} {item_name}")


def _normalize_time(cycle: str, time_value: str) -> str:
    text = str(time_value)
    if cycle == "D":
        return f"{text[:4]}-{text[4:6]}-{text[6:8]}"
    if cycle == "M":
        return f"{text[:4]}-{text[4:6]}-01"
    if cycle == "Q":
        if "Q" in text:
            year, q = text.split("Q", 1)
            month = int(q) * 3
            return f"{year}-{month:02d}-01"
        # Some ECOS quarterly rows are encoded as YYYYMM.
        return f"{text[:4]}-{text[4:6]}-01"
    raise ValueError(f"Unsupported cycle: {cycle}")


def fetch_series(
    stat: str,
    cycle: str,
    item: str,
    start: str,
    end: str,
    api_key: str,
) -> list[tuple[str, float]]:
    url = _ecos_url(["StatisticSearch", api_key, "json", "kr", 1, 100000, stat, cycle, start, end, item])
    data = _get_json(url)
    rows = data.get("StatisticSearch", {}).get("row", [])
    if not rows:
        raise RuntimeError(f"{stat}/{item}: empty response or API error: {json.dumps(data, ensure_ascii=False)[:500]}")

    output: list[tuple[str, float]] = []
    for row in rows:
        value = row.get("DATA_VALUE")
        if value in ("", None):
            continue
        output.append((_normalize_time(cycle, row["TIME"]), float(str(value).replace(",", ""))))
    output.sort(key=lambda pair: pair[0])
    return output


def write_csv(key: str, rows: list[tuple[str, float]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write for {key}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{key}.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["date", "value"])
        writer.writerows(rows)
    print(f"  saved: {path} ({len(rows)} rows, {rows[0][0]} ~ {rows[-1][0]})")


def derive_and_save() -> None:
    import pandas as pd

    def load(key: str) -> pd.Series | None:
        path = OUT_DIR / f"{key}.csv"
        if not path.exists():
            return None
        return pd.read_csv(path, parse_dates=["date"]).set_index("date")["value"].sort_index()

    cpi = load("cpi_index")
    if cpi is not None:
        cpi_yoy = (cpi.pct_change(12) * 100.0).dropna()
        write_csv("cpi_yoy", [(date.strftime("%Y-%m-%d"), round(float(value), 4)) for date, value in cpi_yoy.items()])

    t3 = load("treasury_3y")
    t10 = load("treasury_10y")
    corp = load("corp_aa_3y")
    if t3 is not None and t10 is not None:
        spread = (t10 - t3).dropna()
        write_csv("yield_spread_10y_3y", [(date.strftime("%Y-%m-%d"), round(float(value), 4)) for date, value in spread.items()])
    if t3 is not None and corp is not None:
        credit = (corp - t3).dropna()
        write_csv("credit_spread", [(date.strftime("%Y-%m-%d"), round(float(value), 4)) for date, value in credit.items()])


def _end_for_cycle(cycle: str, end: str) -> str:
    if cycle == "D":
        return end
    if cycle == "M":
        return end[:6]
    if cycle == "Q":
        return f"{end[:4]}Q4"
    raise ValueError(f"Unsupported cycle: {cycle}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key", default=None, help="ECOS API key. Defaults to ECOS_API_KEY env var.")
    parser.add_argument("--discover", metavar="STAT_CODE", help="print StatisticItemList for a statistic code")
    parser.add_argument("--end", default="20261231", help="collection end date, YYYYMMDD")
    parser.add_argument("--allow-partial", action="store_true", help="download configured series even if some SERIES items are still None")
    args = parser.parse_args()

    try:
        api_key = resolve_api_key(args.api_key)
    except RuntimeError as exc:
        print(f"[stop] {exc}")
        return 1

    if args.discover:
        discover(args.discover, api_key)
        return 0

    todo = [key for key, (stat, _cycle, item, _start) in SERIES.items() if stat is None or item is None]
    if todo and not args.allow_partial:
        print(f"[stop] item codes not confirmed: {todo}")
        print("Use: python scripts\\fetch_ecos.py --discover <STAT_CODE>")
        print("Then fill SERIES in scripts/fetch_ecos.py, or pass --allow-partial for configured series only.")
        return 1

    for key, (stat, cycle, item, start) in SERIES.items():
        if stat is None or item is None:
            print(f"skip unconfigured: {key}")
            continue
        print(f"fetch: {key} ({stat}/{cycle}/{item})")
        rows = fetch_series(stat, cycle, item, start, _end_for_cycle(cycle, args.end), api_key)
        write_csv(key, rows)

    derive_and_save()
    print("Done. Next: python tests\\validation_macro_pit.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

