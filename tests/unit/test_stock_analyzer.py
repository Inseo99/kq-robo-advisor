from __future__ import annotations

import numpy as np
import pandas as pd

from kq_tool.analyzer.stock_analyzer import analyze_stock_payload


def _ohlcv(rows: int = 260) -> pd.DataFrame:
    index = pd.date_range("2024-01-02", periods=rows, freq="B")
    close = pd.Series(np.linspace(50_000, 65_000, rows), index=index)
    return pd.DataFrame(
        {
            "Open": close - 100,
            "High": close + 500,
            "Low": close - 500,
            "Close": close,
            "Volume": np.arange(rows) + 1_000,
        },
        index=index,
    )


def test_analyze_stock_payload_preserves_legacy_response_shape() -> None:
    payload = analyze_stock_payload(
        "005930.KS",
        _ohlcv(),
        period="max",
        name="삼성전자",
        info={
            "trailingPE": 12.3,
            "priceToBook": 1.1,
            "returnOnEquity": 0.12,
            "marketCap": 400_000_000_000_000,
        },
    )

    assert payload["ticker"] == "005930.KS"
    assert payload["name"] == "삼성전자"
    assert payload["chart"]["period"] == "max"
    assert payload["chart"]["count"] == 260
    assert {"rsi", "macd", "signal", "atr"}.issubset(payload["indicators"])
    assert {"score", "signal", "cw_score", "cw_signal", "valid_days"}.issubset(payload["robo"])
    assert payload["fund"]["pe"] == 12.3


def test_analyze_stock_payload_can_use_live_price_override() -> None:
    frame = _ohlcv()
    payload = analyze_stock_payload(
        "005930.KS",
        frame,
        current_price=70_000,
        current_source="실시간",
        current_date="2026-06-28",
    )

    assert payload["cur"] == 70_000
    assert payload["cur_source"] == "실시간"
    assert payload["cur_date"] == "2026-06-28"
    assert payload["robo"]["entry"] == 70_000


def test_analyze_stock_payload_refreshes_price_sensitive_fundamentals() -> None:
    payload = analyze_stock_payload(
        "005930.KS",
        _ohlcv(),
        current_price=70_000,
        current_date="2026-06-28",
        info={
            "trailingPE": 7.57,
            "priceToBook": 1.47,
            "returnOnEquity": 0.115,
            "marketCap": 637_500_000_000_000,
            "sharesOutstanding": 1_000,
            "equity": 10_000_000,
            "netIncomeTTM": 5_000_000,
            "fundamentalSource": "excel_pit",
            "fundamentalPeriodDate": "2026-03-31",
            "fundamentalObservableDate": "2026-05-15",
            "returnOnEquity_basis": "latest_reported_quarter",
        },
    )

    assert payload["fund"]["mcap"] == 70_000_000
    assert payload["fund"]["pbr"] == 7.0
    assert payload["fund"]["pe"] == 14.0
    assert payload["fund"]["roe"] == 0.115
    assert payload["fund"]["marketCap_basis"] == "current_price_x_shares"
    assert payload["fund"]["marketCap_date"] == "2026-06-28"
    assert payload["fund"]["fundamental_period_date"] == "2026-03-31"
    assert payload["fund"]["fundamental_observable_date"] == "2026-05-15"
    assert payload["fund"]["roe_basis"] == "latest_reported_quarter"
    assert "현재가" in payload["fund"]["note"]
