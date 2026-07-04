from __future__ import annotations

import numpy as np
import pandas as pd

from kq_tool.screener.engine import (
    build_screener_record,
    latest_price_date_from_groups,
    momentum_12_1,
    momentum_recent,
    optional_float,
    prewarm_screener_cache,
)
from kq_tool.screener.engine import quick_robo_from_close
from kq_tool.screener.strategies import SCREENER_DEFINITIONS, build_screeners, rank_by


def test_optional_float_maps_invalid_and_zero_to_none() -> None:
    assert optional_float(None) is None
    assert optional_float(0) is None
    assert optional_float("12.5") == 12.5


def test_latest_price_date_from_groups_returns_latest_nonempty_date() -> None:
    groups = {
        "A": pd.Series([1.0, 2.0], index=pd.to_datetime(["2026-06-25", "2026-06-26"])),
        "B": pd.Series([None, 3.0], index=pd.to_datetime(["2026-06-24", "2026-06-29"])),
        "C": pd.Series([None, None], index=pd.to_datetime(["2026-06-27", "2026-06-28"])),
    }

    assert latest_price_date_from_groups(groups) == "2026-06-29"


def test_latest_price_date_from_groups_returns_none_without_valid_prices() -> None:
    assert latest_price_date_from_groups({}) is None
    assert latest_price_date_from_groups({"A": pd.Series([None])}) is None


def test_prewarm_screener_cache_loads_price_and_metric_files() -> None:
    loaded: list[str] = []

    result = prewarm_screener_cache(
        loaded.append,
        {"open": "price_open.parquet", "close": "price_close.parquet"},
        {"per": "metric_per.parquet", "eps": "metric_eps.parquet"},
    )

    assert loaded == [
        "price_open.parquet",
        "price_close.parquet",
        "metric_per.parquet",
        "metric_eps.parquet",
    ]
    assert result["loaded"] == loaded
    assert result["failed"] == []


def test_prewarm_screener_cache_records_failures_and_continues() -> None:
    calls: list[str] = []

    def loader(fname: str) -> None:
        calls.append(fname)
        if fname == "bad.parquet":
            raise RuntimeError("load failed")

    result = prewarm_screener_cache(
        loader,
        {"open": "bad.parquet", "close": "ok.parquet"},
        {},
    )

    assert calls == ["bad.parquet", "ok.parquet"]
    assert result["loaded"] == ["ok.parquet"]
    assert result["failed"] == ["bad.parquet"]


def test_momentum_12_1_requires_enough_history() -> None:
    assert momentum_12_1(pd.Series([1, 2, 3])) is None


def test_momentum_12_1_excludes_latest_month() -> None:
    close = pd.Series(np.linspace(100, 150, 260))
    close.iloc[-252] = 100
    close.iloc[-60] = 200
    close.iloc[-21] = 180
    close.iloc[-1] = 90

    assert momentum_12_1(close) == 0.8
    assert momentum_recent(close) == -0.55


def test_momentum_recent_uses_latest_price() -> None:
    close = pd.Series(np.linspace(100, 130, 80))

    assert momentum_recent(close) == close.iloc[-1] / close.iloc[-60] - 1


def test_rank_by_handles_low_and_high_rules() -> None:
    results = {
        "A": {"pe": 8, "roe": 0.1},
        "B": {"pe": 5, "roe": 0.2},
        "C": {"pe": None, "roe": -0.1},
    }

    assert rank_by(results, "pe") == ["B", "A"]
    assert rank_by(results, "roe", reverse=True) == ["B", "A", "C"]


def test_build_screeners_returns_all_default_lists() -> None:
    results = {
        "A": {"pe": 8, "pbr": 1.0, "roe": 0.1, "mom": 0.2, "s2_mom": 0.1, "dcf_g": -0.1, "score": 70},
        "B": {"pe": 5, "pbr": 2.0, "roe": 0.2, "mom": 0.1, "s2_mom": 0.3, "dcf_g": 0.1, "score": 40},
    }

    screeners = build_screeners(results, limit=1)

    assert screeners["저PER"] == ["B"]
    assert screeners["고ROE"] == ["B"]
    assert screeners["모멘텀"] == ["A"]
    assert screeners["S2모멘텀"] == ["B"]
    assert screeners["로보매수"] == ["A"]


def test_screener_definition_order_matches_ui_order() -> None:
    assert list(SCREENER_DEFINITIONS)[:6] == [
        "저PER",
        "저PBR",
        "고ROE",
        "모멘텀",
        "역방향DCF",
        "S2모멘텀",
    ]


def test_build_screener_record_matches_expected_shape() -> None:
    index = pd.date_range("2024-01-01", periods=260, freq="B")
    close = pd.Series(np.linspace(100, 130, len(index)), index=index)
    info = {"trailingPE": 10, "priceToBook": 1.2, "returnOnEquity": 0.15, "trailingEps": 8}

    record = build_screener_record("005930.KS", "삼성전자", close, info, required_return=0.095)

    assert record["name"] == "삼성전자"
    assert record["pe"] == 10
    assert record["pbr"] == 1.2
    assert record["roe"] == 0.15
    assert record["mom"] is not None
    assert record["s2_mom"] is not None
    assert record["signal"] in {"매수", "관망", "매도"}


def test_quick_robo_from_close_uses_shared_thresholds() -> None:
    index = pd.date_range("2024-01-01", periods=80, freq="B")
    close = pd.Series(np.r_[np.linspace(100, 70, 40), np.linspace(70, 110, 40)], index=index)

    score, signal = quick_robo_from_close(close)

    assert 0 <= score <= 100
    assert signal in {"매수", "관망", "매도"}


def test_server_reuses_screener_price_date_helper() -> None:
    import server

    assert server._kq_latest_price_date_from_groups is latest_price_date_from_groups
    assert server._kq_prewarm_screener_cache is prewarm_screener_cache
