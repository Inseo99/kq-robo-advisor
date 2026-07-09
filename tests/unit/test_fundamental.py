from __future__ import annotations

from kq_tool.data.fundamental import (
    MCAP_FIN_KEY,
    excel_fundamental_info,
    has_yfinance_fundamental_info,
    metric_or_calc,
    refresh_market_sensitive_fundamentals,
    sample_fundamental_info,
)


def test_metric_or_calc_prefers_direct_metric() -> None:
    assert metric_or_calc(10.5, 20.0, 0, 200) == 10.5


def test_metric_or_calc_uses_calculated_fallback() -> None:
    assert metric_or_calc(None, 15.123, 0, 200) == 15.12


def test_metric_or_calc_rejects_out_of_range_values() -> None:
    assert metric_or_calc(500, None, 0, 200) is None


def test_excel_fundamental_info_builds_yfinance_style_dict() -> None:
    latest = {
        MCAP_FIN_KEY: 1_000,
        "당기순이익(천원)": 25_000,
        "자본총계(천원)": 500_000,
        "기말발행주식수(보통주)(주)": 1_000,
        "ROE(%)": 12.3,
        "영업활동으로인한현금흐름(천원)": 30_000,
    }

    metrics = {"per": None, "pbr": None, "eps": 100, "bps": 500, "div_yield": 2.5}
    info = excel_fundamental_info(latest, lambda key: metrics.get(key))

    assert info["marketCap"] == 1_000_000_000.0
    assert info["trailingPE"] == 10.0
    assert info["priceToBook"] == 2.0
    assert info["equity"] == 500_000_000.0
    assert info["netIncomeTTM"] == 100_000_000.0
    assert info["marketCap_basis"] == "pit_financial_snapshot"
    assert info["trailingEps"] == 100
    assert info["dividendYield"] == 0.025


def test_excel_fundamental_info_prefers_latest_snapshot_metadata() -> None:
    latest = {
        MCAP_FIN_KEY: 1_000,
        "당기순이익(천원)": 25_000,
        "자본총계(천원)": 500_000,
        "기말발행주식수(보통주)(주)": 1_000,
        "ROE(%)": 12.3,
        "__market_cap_latest": 1_500,
        "__net_income_ttm": 120_000,
        "__equity_latest": 600_000,
        "__shares_latest": 2_000,
        "__roe_latest": 18.5,
        "__latest_period_date": "2026-03-31",
        "__latest_observable_date": "2026-05-15",
    }

    info = excel_fundamental_info(latest, lambda _key: None)

    assert info["marketCap"] == 1_500_000_000.0
    assert info["trailingPE"] == 12.5
    assert info["priceToBook"] == 2.5
    assert info["returnOnEquity"] == 0.185
    assert info["returnOnEquity_basis"] == "latest_reported_quarter"
    assert info["sharesOutstanding"] == 2_000
    assert info["equity"] == 600_000_000.0
    assert info["netIncomeTTM"] == 120_000_000.0
    assert info["fundamentalPeriodDate"] == "2026-03-31"
    assert info["fundamentalObservableDate"] == "2026-05-15"


def test_refresh_market_sensitive_fundamentals_uses_current_price() -> None:
    info = {
        "trailingPE": 7.57,
        "priceToBook": 1.47,
        "marketCap": 637_500,
        "sharesOutstanding": 100,
        "equity": 1_000,
        "netIncomeTTM": 50,
        "returnOnEquity": 0.115,
        "fundamentalSource": "excel_pit",
    }

    refreshed = refresh_market_sensitive_fundamentals(
        info,
        current_price=20,
        current_date="2026-07-07",
    )

    assert refreshed["marketCap"] == 2_000
    assert refreshed["priceToBook"] == 2.0
    assert refreshed["trailingPE"] == 40.0
    assert refreshed["returnOnEquity"] == 0.115
    assert refreshed["marketCap_basis"] == "current_price_x_shares"
    assert refreshed["marketCap_date"] == "2026-07-07"
    assert "현재가" in refreshed["fundamental_note"]


def test_refresh_market_sensitive_fundamentals_keeps_snapshot_when_inputs_missing() -> None:
    info = {"marketCap": 637_500, "sharesOutstanding": None}

    refreshed = refresh_market_sensitive_fundamentals(info, current_price=20)

    assert refreshed["marketCap"] == 637_500
    assert "재계산을 적용하지 못했습니다" in refreshed["fundamental_note"]


def test_has_yfinance_fundamental_info_requires_mapping_and_required_key() -> None:
    assert has_yfinance_fundamental_info({"trailingPE": 10.5}) is True
    assert has_yfinance_fundamental_info({"trailingPE": None}) is False
    assert has_yfinance_fundamental_info({"priceToBook": 1.2}) is False
    assert has_yfinance_fundamental_info(None) is False


def test_sample_fundamental_info_is_deterministic_for_ticker() -> None:
    assert sample_fundamental_info("005930.KS") == sample_fundamental_info("005930.KS")


def test_server_reuses_fundamental_info_helper() -> None:
    import server

    assert server._kq_has_yfinance_fundamental_info is has_yfinance_fundamental_info
