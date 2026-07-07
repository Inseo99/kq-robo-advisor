"""Lightweight API smoke checks for a running KQ Quant Tool server."""

from __future__ import annotations

import argparse
import json
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen


def fetch_json(base_url: str, path: str, timeout: float = 30.0) -> dict:
    url = base_url.rstrip("/") + path
    with urlopen(url, timeout=timeout) as response:
        body = response.read().decode("utf-8")
    return json.loads(body)


def strategy_backtest_check(
    name: str,
    payload: dict,
    *,
    expected_strategy: str | None = None,
    require_benchmark: bool = False,
) -> tuple[str, bool, dict]:
    ok = "error" not in payload and bool(payload.get("metrics"))
    if expected_strategy is not None:
        ok = ok and payload.get("strategy") == expected_strategy
    if require_benchmark:
        ok = ok and bool(payload.get("benchmark"))
    return (
        name,
        ok,
        {
            "strategy": payload.get("strategy"),
            "n_rebalance": payload.get("n_rebalance"),
            "metrics": bool(payload.get("metrics")),
            "benchmark": bool(payload.get("benchmark")),
            "costs": bool(payload.get("costs")),
        },
    )


def quant_compare_check(payload: dict) -> tuple[str, bool, dict]:
    runs = payload.get("comparison_runs", [])
    benchmark = payload.get("comparison_benchmark", {})
    ok = (
        "error" not in payload
        and payload.get("strategy") == "quant_compare"
        and len(runs) >= 4
        and bool(payload.get("overlay_report", {}).get("reports"))
        and bool(benchmark.get("equity"))
    )
    return (
        "stratbt_compare",
        ok,
        {
            "runs": [run.get("label") for run in runs],
            "overlay_report": bool(payload.get("overlay_report", {}).get("reports")),
            "benchmark": bool(benchmark.get("equity")),
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8888")
    parser.add_argument("--ticker", default="005930.KS")
    parser.add_argument("--period", default="1mo")
    parser.add_argument("--include-stock", action="store_true")
    parser.add_argument("--include-core", action="store_true")
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args()

    checks = []
    try:
        ping = fetch_json(args.base_url, "/api/ping", timeout=args.timeout)
        checks.append(("ping", ping.get("ok") is True, ping))

        health = fetch_json(args.base_url, "/api/health", timeout=args.timeout)
        checks.append(("health", health.get("ok") is True, health))

        if args.include_stock:
            query = urlencode({"t": args.ticker, "p": args.period})
            stock = fetch_json(args.base_url, f"/api/stock?{query}", timeout=args.timeout)
            chart = stock.get("chart", {})
            stock_ok = "error" not in stock and bool(chart.get("dates"))
            checks.append(
                (
                    "stock",
                    stock_ok,
                    {"ticker": stock.get("ticker"), "chart_count": len(chart.get("dates", []))},
                )
            )

        if args.include_core:
            screen = fetch_json(args.base_url, "/api/screen", timeout=args.timeout)
            screeners = screen.get("screeners", {})
            screen_ok = (
                "error" not in screen
                and bool(screen.get("data"))
                and "S2모멘텀" in screeners
            )
            checks.append(
                (
                    "screen",
                    screen_ok,
                    {
                        "count": len(screen.get("data", {})),
                        "price_date": screen.get("price_date"),
                        "s2": len(screeners.get("S2모멘텀", [])),
                    },
                )
            )

            strat_query = urlencode({"s": "quant", "n": 3, "r": "Q", "p": "1y", "tc": 10, "slip": 5})
            strat = fetch_json(args.base_url, f"/api/stratbt?{strat_query}", timeout=args.timeout)
            checks.append(strategy_backtest_check("stratbt", strat, expected_strategy="quant"))

            s2_query = urlencode({"s": "quant_s2", "n": 3, "r": "Q", "p": "1y", "tc": 10, "slip": 5})
            s2 = fetch_json(args.base_url, f"/api/stratbt?{s2_query}", timeout=args.timeout)
            checks.append(
                strategy_backtest_check(
                    "stratbt_s2",
                    s2,
                    expected_strategy="quant_s2",
                    require_benchmark=True,
                )
            )

            compare_query = urlencode({"s": "quant_compare", "n": 3, "r": "Q", "p": "1y", "tc": 10, "slip": 5})
            compare = fetch_json(args.base_url, f"/api/stratbt?{compare_query}", timeout=args.timeout)
            checks.append(quant_compare_check(compare))

            recommend = fetch_json(args.base_url, "/api/recommend_portfolio", timeout=args.timeout)
            recommend_ok = "error" not in recommend and bool(recommend.get("weights"))
            checks.append(
                (
                    "recommend",
                    recommend_ok,
                    {
                        "assets": len(recommend.get("assets", [])),
                        "validity": bool(recommend.get("validity")),
                    },
                )
            )
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
        print(f"SMOKE FAIL: {exc}")
        return 1

    failed = False
    for name, ok, payload in checks:
        status = "OK" if ok else "FAIL"
        print(f"{status:4s} {name}: {payload}")
        failed = failed or not ok
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
