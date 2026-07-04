r"""Search ECOS statistic tables by keyword.

Use this before ``scripts/fetch_ecos.py --discover <STAT_CODE>`` when the
statistic table code is unknown.

Examples
--------
    python scripts\search_ecos.py 경기종합지수
    python scripts\search_ecos.py 산업생산
    python scripts\search_ecos.py 수출입
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.parse
import urllib.request
from typing import Iterable

try:
    from fetch_ecos import API_KEY  # same directory when run as scripts/search_ecos.py
except Exception:  # pragma: no cover - defensive fallback for direct imports
    API_KEY = ""

BASE = "https://ecos.bok.or.kr/api"


def resolve_api_key(cli_key: str | None = None) -> str:
    key = cli_key or os.getenv("ECOS_API_KEY") or API_KEY
    if not key or key.startswith("여기에"):
        raise RuntimeError(
            "ECOS API key is missing. Set ECOS_API_KEY or pass --api-key."
        )
    return key


def _get_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def _ecos_url(path_parts: Iterable[str | int]) -> str:
    encoded = [urllib.parse.quote(str(part), safe="") for part in path_parts]
    return "/".join([BASE, *encoded])


def _norm(value: object) -> str:
    return str(value or "").casefold().replace(" ", "")


def search_tables(
    api_key: str,
    keywords: list[str],
    page_size: int = 1000,
    max_pages: int = 20,
) -> list[dict]:
    terms = [_norm(keyword) for keyword in keywords if str(keyword).strip()]
    if not terms:
        raise ValueError("At least one keyword is required.")

    matches: list[dict] = []
    start = 1
    pages = 0
    total = None

    while pages < max_pages:
        end = start + page_size - 1
        url = _ecos_url(["StatisticTableList", api_key, "json", "kr", start, end])
        data = _get_json(url)
        block = data.get("StatisticTableList") or {}
        rows = block.get("row") or []
        if not rows:
            if pages == 0:
                print(f"No rows or API error: {json.dumps(data, ensure_ascii=False)[:500]}")
            break

        if total is None:
            try:
                total = int(block.get("list_total_count") or len(rows))
            except Exception:
                total = len(rows)

        for row in rows:
            haystack = _norm(" ".join(str(v) for v in row.values()))
            if all(term in haystack for term in terms):
                matches.append(row)

        pages += 1
        if total is not None and end >= total:
            break
        start = end + 1

    return matches


def print_matches(matches: list[dict]) -> None:
    if not matches:
        print("No matching statistic tables found.")
        return

    print(f"matches: {len(matches)}")
    for row in matches:
        stat_code = row.get("STAT_CODE") or row.get("STAT_CODE1") or ""
        stat_name = row.get("STAT_NAME") or row.get("STAT_NAME1") or ""
        cycle = row.get("CYCLE") or ""
        org_name = row.get("ORG_NAME") or ""
        srch = row.get("SRCH_YN") or ""
        print(f"{stat_code:<10} cycle={cycle:<3} srch={srch:<2} {stat_name} {org_name}")
        if stat_code:
            print(f"  discover: python scripts\\fetch_ecos.py --discover {stat_code}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("keywords", nargs="+", help="keyword(s) to search in ECOS table names")
    parser.add_argument("--api-key", default=None, help="ECOS API key. Defaults to ECOS_API_KEY env var.")
    parser.add_argument("--page-size", type=int, default=1000)
    parser.add_argument("--max-pages", type=int, default=20)
    args = parser.parse_args()

    try:
        api_key = resolve_api_key(args.api_key)
        matches = search_tables(api_key, args.keywords, args.page_size, args.max_pages)
    except Exception as exc:
        print(f"[stop] {exc}")
        return 1

    print_matches(matches)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
