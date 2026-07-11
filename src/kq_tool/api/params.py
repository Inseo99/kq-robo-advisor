"""API query parameter parsing helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


QueryParams = Mapping[str, list[str]]


@dataclass(frozen=True)
class StockParams:
    ticker: str
    period: str


@dataclass(frozen=True)
class StrategyBacktestParams:
    strategy: str
    top_n: int
    rebalance: str
    period: str
    transaction_cost_bps: float
    slippage_bps: float


def first_query_value(query: QueryParams, key: str, default: str) -> str:
    """Return the first query value, preserving legacy parse_qs semantics."""

    values = query.get(key)
    if not values:
        return default
    return values[0]


def parse_stock_params(query: QueryParams) -> StockParams:
    """Parse `/api/stock` query parameters."""

    return StockParams(
        ticker=first_query_value(query, "t", "005930.KS").strip(),
        period=first_query_value(query, "p", "1y"),
    )


def parse_strategy_backtest_params(query: QueryParams) -> StrategyBacktestParams:
    """Parse `/api/stratbt` query parameters, including trading frictions."""

    return StrategyBacktestParams(
        strategy=first_query_value(query, "s", "s1m_lsv"),
        top_n=int(first_query_value(query, "n", "5")),
        rebalance=first_query_value(query, "r", "M"),
        period=first_query_value(query, "p", "3y"),
        transaction_cost_bps=float(first_query_value(query, "tc", "10")),
        slippage_bps=float(first_query_value(query, "slip", "5")),
    )

