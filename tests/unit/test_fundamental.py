from __future__ import annotations

from kq_tool.data.fundamental import (
    MCAP_FIN_KEY,
    excel_fundamental_info,
    has_yfinance_fundamental_info,
    metric_or_calc,
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
    assert info["trailingEps"] == 100
    assert info["dividendYield"] == 0.025


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
