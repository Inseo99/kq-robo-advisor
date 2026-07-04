from __future__ import annotations

import pytest

from kq_tool.validation.segments import (
    segment_split_for_top,
    signal_names_for_side,
    ticker_segment,
)


SIGNALS = {
    "buy_rsi": (+1, "s_rsi"),
    "sell_rsi": (-1, "s_rsi"),
    "buy_macd": (+1, "s_macd"),
}


def test_signal_names_for_side_filters_buy_sell() -> None:
    assert signal_names_for_side(SIGNALS, "buy") == ["buy_rsi", "buy_macd"]
    assert signal_names_for_side(SIGNALS, "sell") == ["sell_rsi"]
    assert signal_names_for_side(SIGNALS, "all") == ["buy_rsi", "sell_rsi", "buy_macd"]


def test_signal_names_for_side_rejects_unknown_side() -> None:
    with pytest.raises(ValueError):
        signal_names_for_side(SIGNALS, "long_only")


def test_ticker_segment_splits_ranked_universe() -> None:
    tickers = ["A", "B", "C", "D"]

    assert ticker_segment(tickers, "large", 2) == ["A", "B"]
    assert ticker_segment(tickers, "mid_small", 2) == ["C", "D"]
    assert ticker_segment(tickers, "all", 2) == tickers


def test_segment_split_for_top_defaults_to_half_and_clamps() -> None:
    assert segment_split_for_top(100) == 50
    assert segment_split_for_top(5) == 2
    assert segment_split_for_top(5, 99) == 4
    assert segment_split_for_top(1) == 1
