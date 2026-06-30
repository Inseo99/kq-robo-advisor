from __future__ import annotations

from kq_tool.data.universe import FALLBACK_UNIVERSE, build_universe, market_counts, yahoo_suffix


def test_yahoo_suffix_maps_kospi_to_ks() -> None:
    assert yahoo_suffix("KOSPI") == ".KS"
    assert yahoo_suffix("KOSDAQ") == ".KQ"


def test_build_universe_uses_fallback_when_excel_missing() -> None:
    universe = build_universe(None)

    assert universe == FALLBACK_UNIVERSE
    assert "005930.KS" in universe


def test_build_universe_converts_excel_rows_to_yahoo_tickers() -> None:
    excel_data = {
        "005930": {"market": "KOSPI", "name": "삼성전자", "sector": "반도체"},
        "123456": {"market": "KOSDAQ", "name": "테스트", "sector": None},
    }

    universe = build_universe(excel_data)

    assert universe["005930.KS"] == ("삼성전자", "반도체")
    assert universe["123456.KQ"] == ("테스트", "기타")


def test_market_counts_counts_suffixes() -> None:
    counts = market_counts({"005930.KS": (), "000660.KS": (), "123456.KQ": ()})

    assert counts == {"KOSPI": 2, "KOSDAQ": 1}
