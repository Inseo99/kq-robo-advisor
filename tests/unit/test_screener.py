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
from kq_tool.screener.strategies import (
    DISPLAY_NAME_QUALITY,
    SCREENER_DEFINITIONS,
    build_screeners,
    kang_mode,
    rank_by,
    screener_diagnostics,
    screener_metadata,
)


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
        "A": {
            "pe": 8,
            "pbr": 1.2,
            "roe": 0.1,
            "mom": 0.2,
            "s2_mom": 0.1,
            "mcap": 100.0,
            "free_cashflow": 10.0,
            "net_income_ttm": 10.0,
            "quality": 0.1,
        },
        "B": {
            "pe": 5,
            "pbr": 0.8,
            "roe": 0.2,
            "mom": 0.1,
            "s2_mom": 0.3,
            "mcap": 50.0,
            "free_cashflow": 10.0,
            "net_income_ttm": 10.0,
            "quality": 0.2,
        },
    }

    screeners = build_screeners(results, limit=1)

    assert screeners["s1m_lsv"] == ["B"]
    assert screeners["s2m_lsv"] == ["B"]
    assert screeners["s3m_lsv"] == ["B"]
    assert screeners["s1m_kang"] == ["B"]
    assert screeners["s2m_kang"] == ["B"]
    assert screeners["s3m_kang"] == ["B"]


def test_screener_definition_order_matches_ui_order() -> None:
    assert list(SCREENER_DEFINITIONS)[:6] == [
        "s1m_lsv",
        "s2m_lsv",
        "s3m_lsv",
        "s1m_kang",
        "s2m_kang",
        "s3m_kang",
    ]


def test_screener_metadata_exposes_six_backtest_tabs() -> None:
    meta = screener_metadata()

    assert [m["key"] for m in meta] == list(SCREENER_DEFINITIONS)
    assert meta[0]["label"] == "M1 LSV"
    assert meta[3]["label"] == f"M1 {DISPLAY_NAME_QUALITY}"
    assert "위험회피" not in meta[0]["selection"]
    assert "t1" in meta[1]["gate"]
    assert "t2" in meta[2]["gate"]


def test_kang_screener_falls_back_to_small_cap_positive_roe() -> None:
    results = {
        "A": {"mcap": 100.0, "quality": 0.3},
        "B": {"mcap": 50.0, "quality": 0.2},
        "C": {"mcap": 40.0, "quality": -0.1},
        "D": {"mcap": 10.0, "quality": 0.05},
    }

    screeners = build_screeners(results, limit=2)

    assert screeners["s1m_kang"] == ["B", "D"]


def test_kang_fallback_mode_is_exposed_in_diagnostics() -> None:
    results = {
        "A": {"mcap": 100.0, "quality": 0.3, "pe": 8.0, "pbr": 1.0, "sector": "IT"},
        "B": {"mcap": 50.0, "quality": 0.2, "pe": 5.0, "pbr": 0.8, "sector": "금융"},
        "C": {"mcap": 40.0, "quality": -0.1, "pe": 4.0, "pbr": 0.7, "sector": ""},
        "D": {"mcap": 10.0, "quality": 0.05, "pe": None, "pbr": 0.5, "sector": "소재"},
    }
    screeners = build_screeners(results, limit=2)
    diagnostics = screener_diagnostics(results, screeners)

    assert kang_mode(results) == "fallback_roe"
    assert diagnostics["kang_mode"] == "fallback_roe"
    assert diagnostics["sector_mapped_count"] == 3
    assert diagnostics["per_filter_count"]["lsv_positive_pe_pbr"] == 3
    assert diagnostics["per_filter_count"]["kang_strict_cashflow_income"] == 0
    assert diagnostics["per_filter_count"]["kang_fallback_positive_roe"] == 3
    assert diagnostics["per_filter_count"]["kang_active_pool"] == 3
    assert diagnostics["per_filter_count"]["kang_small_cap_bucket"] == 2
    assert diagnostics["per_filter_count"]["kang_final_selected"] == 2
    assert "fallback" in diagnostics["warnings"][0]


def test_kang_strict_mode_takes_precedence_over_fallback() -> None:
    results = {
        "A": {
            "mcap": 100.0,
            "quality": 0.3,
            "free_cashflow": 10.0,
            "net_income_ttm": 20.0,
            "operating_cashflow": 10.0,
            "net_income_common": 20.0,
            "total_assets": 100.0,
            "operating_income": 15.0,
        },
        "B": {"mcap": 50.0, "quality": 0.2},
    }
    screeners = build_screeners(results, limit=2)
    diagnostics = screener_diagnostics(results, screeners)

    assert kang_mode(results) == "strict_yfinance"
    assert diagnostics["kang_mode"] == "strict_yfinance"
    assert diagnostics["per_filter_count"]["kang_strict_cashflow_income"] == 1
    assert diagnostics["per_filter_count"]["kang_strict_yfinance"] == 1
    assert diagnostics["per_filter_count"]["kang_small_cap_bucket"] == 1
    assert diagnostics["per_filter_count"]["kang_final_selected"] == 1
    assert diagnostics["warnings"] == []


def test_build_screener_record_matches_expected_shape() -> None:
    index = pd.date_range("2024-01-01", periods=260, freq="B")
    close = pd.Series(np.linspace(100, 130, len(index)), index=index)
    info = {
        "trailingPE": 10,
        "priceToBook": 1.2,
        "returnOnEquity": 0.15,
        "trailingEps": 8,
        "marketCap": 1_000_000_000_000,
        "sharesOutstanding": 1_000_000,
        "freeCashflow": 10_000_000_000,
        "operatingCashflow": 10_000_000_000,
        "netIncomeToCommon": 20_000_000_000,
        "totalAssets": 200_000_000_000,
        "operatingIncome": 30_000_000_000,
        "netIncomeTTM": 20_000_000_000,
        "equity": 100_000_000_000,
        "kangStrictSource": "yf_cache",
    }

    record = build_screener_record("005930.KS", "삼성전자", close, info, required_return=0.095)

    assert record["name"] == "삼성전자"
    assert record["mcap"] == 130_000_000.0
    assert record["mcap_basis"] == "current_price_x_shares"
    assert record["pe"] == 10
    assert record["pbr"] == 1.2
    assert record["roe"] == 0.15
    assert record["free_cashflow"] == 10_000_000_000
    assert record["net_income_ttm"] == 20_000_000_000
    assert record["operating_cashflow"] == 10_000_000_000
    assert record["net_income_common"] == 20_000_000_000
    assert record["total_assets"] == 200_000_000_000
    assert record["operating_income"] == 30_000_000_000
    assert record["kang_strict_source"] == "yf_cache"
    assert record["quality"] == 0.15
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



