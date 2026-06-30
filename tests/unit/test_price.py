from __future__ import annotations

import pandas as pd

from kq_tool.data.price import (
    PERIOD_DAYS,
    days_for_period,
    extract_live_price,
    filter_price_period,
    has_min_rows_for_period,
    has_price_history,
    min_rows_for_period,
    normalize_yfinance_columns,
    prepare_yfinance_price_frame,
    resolve_current_price_context,
    sample_price,
    warm_yfinance_session,
)


def _price_frame(rows: int = 300) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=rows, freq="B")
    return pd.DataFrame({"Close": range(rows)}, index=index)


def test_min_rows_for_period_allows_single_day_chart() -> None:
    assert min_rows_for_period("1d") == 1
    assert min_rows_for_period("1mo") == 20


def test_days_for_period_uses_shared_lookup_with_default() -> None:
    assert days_for_period("1mo") == PERIOD_DAYS["1mo"]
    assert days_for_period("7y") == PERIOD_DAYS["7y"]
    assert days_for_period("unknown", default_days=123) == 123
    assert days_for_period(None, default_days=456) == 456


def test_filter_price_period_anchors_to_last_available_date() -> None:
    frame = _price_frame()

    filtered = filter_price_period(frame, "1mo")

    assert filtered.index[-1] == frame.index[-1]
    assert filtered.index[0] >= frame.index[-1] - pd.Timedelta(days=30)


def test_filter_price_period_returns_original_when_too_short() -> None:
    frame = _price_frame(rows=5)

    assert filter_price_period(frame, "1mo").equals(frame)


def test_normalize_yfinance_columns_flattens_multiindex() -> None:
    frame = pd.DataFrame(
        [[1, 2]],
        columns=pd.MultiIndex.from_tuples([("Close", "005930.KS"), ("Open", "005930.KS")]),
    )

    normalized = normalize_yfinance_columns(frame)

    assert list(normalized.columns) == ["Close", "Open"]


def test_prepare_yfinance_price_frame_normalizes_and_validates_rows() -> None:
    frame = pd.DataFrame(
        [[1, 2]] * 20,
        columns=pd.MultiIndex.from_tuples([("Close", "005930.KS"), ("Open", "005930.KS")]),
    )

    prepared = prepare_yfinance_price_frame(frame, "1mo")

    assert prepared is not None
    assert list(prepared.columns) == ["Close", "Open"]


def test_prepare_yfinance_price_frame_rejects_short_non_daily_frames() -> None:
    assert prepare_yfinance_price_frame(_price_frame(rows=1), "1mo") is None
    assert prepare_yfinance_price_frame(_price_frame(rows=1), "1d") is not None


def test_extract_live_price_uses_yfinance_priority_order() -> None:
    assert extract_live_price(
        {
            "currentPrice": None,
            "regularMarketPrice": "70500",
            "previousClose": 70000,
        }
    ) == 70500.0
    assert extract_live_price({"currentPrice": 71000, "regularMarketPrice": 70500}) == 71000.0


def test_extract_live_price_rejects_invalid_values() -> None:
    assert extract_live_price(None) is None
    assert extract_live_price({"currentPrice": -1, "regularMarketPrice": "bad"}) is None


def test_resolve_current_price_context_uses_excel_context_without_live_price() -> None:
    index = pd.date_range("2026-06-25", periods=2, freq="B")

    context = resolve_current_price_context(index, {"currentPrice": None})

    assert context == {"price": None, "source": "엑셀", "date": "2026-06-26"}


def test_resolve_current_price_context_uses_live_price_and_today_when_available() -> None:
    index = pd.date_range("2026-06-25", periods=2, freq="B")

    context = resolve_current_price_context(
        index,
        {"regularMarketPrice": 70500},
        now=pd.Timestamp("2026-06-30"),
    )

    assert context == {"price": 70500.0, "source": "실시간", "date": "2026-06-30"}


def test_has_price_history_detects_nonempty_frames() -> None:
    assert has_price_history(_price_frame(rows=1)) is True
    assert has_price_history(pd.DataFrame()) is False
    assert has_price_history(None) is False


def test_has_min_rows_for_period_uses_period_specific_thresholds() -> None:
    assert has_min_rows_for_period(_price_frame(rows=1), "1d") is True
    assert has_min_rows_for_period(_price_frame(rows=1), "1mo") is False
    assert has_min_rows_for_period(_price_frame(rows=20), "1mo") is True
    assert has_min_rows_for_period(pd.DataFrame(), "1d") is False


def test_warm_yfinance_session_uses_injected_history_loader() -> None:
    calls: list[tuple[str, str]] = []

    def loader(ticker: str, period: str) -> pd.DataFrame:
        calls.append((ticker, period))
        return _price_frame(rows=3)

    assert warm_yfinance_session("005930.KS", "5d", history_loader=loader) is True
    assert calls == [("005930.KS", "5d")]


def test_warm_yfinance_session_returns_false_for_loader_failure() -> None:
    def loader(ticker: str, period: str) -> pd.DataFrame:
        raise RuntimeError("network down")

    assert warm_yfinance_session(history_loader=loader) is False


def test_server_reuses_live_price_extractor() -> None:
    import server

    assert server._kq_days_for_period is days_for_period
    assert server._kq_extract_live_price is extract_live_price
    assert server._kq_has_min_rows_for_period is has_min_rows_for_period
    assert server._kq_prepare_yfinance_price_frame is prepare_yfinance_price_frame
    assert server._kq_resolve_current_price_context is resolve_current_price_context
    assert server._kq_warm_yfinance_session is warm_yfinance_session
    assert server._period_days("1mo") == days_for_period("1mo")
    assert server._period_days("unknown", default_days=456) == 456


def test_sample_price_has_ohlcv_columns() -> None:
    frame = sample_price("005930.KS", n=10, end=pd.Timestamp("2024-01-31"))

    assert list(frame.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert len(frame) == 10
    assert frame.index[-1] == pd.Timestamp("2024-01-31")
