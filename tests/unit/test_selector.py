from __future__ import annotations

import numpy as np
import pandas as pd

from kq_tool.backtest.selector import (
    build_robo_precomputed_indicators,
    latest_at_or_before,
    robo_scores_from_indicators,
    robo_filter_decision,
    select_robo_filtered_momentum,
    select_for_backtest,
    select_momentum,
    select_robo,
    select_s2_momentum,
)


def _hist(rows: int = 80) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=rows, freq="B")
    x = np.arange(rows)
    return pd.DataFrame(
        {
            "A": 100 + x * 0.30 + np.sin(x / 3) * 2.0,
            "B": 120 - x * 0.20 + np.sin(x / 4) * 2.0,
            "C": 100 + x * 0.10 + np.cos(x / 5) * 2.0,
        },
        index=index,
    )


def _momentum_split_hist() -> pd.DataFrame:
    rows = 260
    index = pd.date_range("2024-01-01", periods=rows, freq="B")

    def path(points: list[tuple[int, float]]) -> np.ndarray:
        xs, ys = zip(*points)
        return np.interp(np.arange(rows), xs, ys)

    return pd.DataFrame(
        {
            # Strong recent 3-month momentum, weaker 12-1 momentum.
            "A": path([(0, 100), (8, 100), (200, 100), (239, 110), (259, 150)]),
            # Weak recent momentum, strong S2 12-1 momentum.
            "B": path([(0, 100), (8, 100), (200, 200), (239, 190), (259, 180)]),
            "C": path([(0, 100), (259, 100)]),
        },
        index=index,
    )


def test_latest_at_or_before_returns_prior_row() -> None:
    frame = _hist(5)

    row = latest_at_or_before(frame, frame.index[2] + pd.Timedelta(hours=12))

    assert row.equals(frame.iloc[2])


def test_select_momentum_uses_fallback_when_less_than_252_rows() -> None:
    assert select_momentum(_hist(80), 1) == ["A"]


def test_select_s2_momentum_requires_full_12_1_window() -> None:
    assert select_s2_momentum(_hist(80), 1) == []
    assert select_for_backtest(_hist(260), "quant_s2", 1)


def test_recent_and_s2_momentum_can_select_different_tickers() -> None:
    hist = _momentum_split_hist()

    assert select_momentum(hist, 1) == ["A"]
    assert select_s2_momentum(hist, 1) == ["B"]
    assert select_for_backtest(hist, "quant", 1) == ["A"]
    assert select_for_backtest(hist, "quant_s2", 1) == ["B"]


def test_robo_scores_from_indicators_aligns_labels_after_filtering() -> None:
    hist = _hist(80)[["A", "C"]]
    rsi_last = pd.Series({"A": 20, "B": 80, "C": 50})
    macd_bull = pd.Series({"A": True, "B": False, "C": True})
    ma20 = pd.Series({"A": 100, "B": 100, "C": 100})
    ma60 = pd.Series({"A": 100, "B": 100, "C": 100})
    cur = hist.iloc[-1]

    scores = robo_scores_from_indicators(hist, rsi_last, macd_bull, ma20, ma60, cur)

    assert list(scores.index) == ["A", "C"]
    assert "B" not in scores.index


def test_select_robo_returns_top_positive_scores() -> None:
    hist = _hist(80)

    selected = select_robo(hist, top_n=2)

    assert len(selected) == 2


def test_select_robo_filtered_momentum_replaces_bearish_top_candidate() -> None:
    rows = 80
    index = pd.date_range("2024-01-01", periods=rows, freq="B")
    x = np.arange(rows)
    hist = pd.DataFrame(
        {
            "A": 100 + x * 1.00,
            "B": 100 + x * 0.70,
            "C": 100 + x * 0.50,
            "D": 100 + x * 0.20,
        },
        index=index,
    )
    cur_date = hist.index[-1]
    precomputed = {
        "rsi": pd.DataFrame([{"A": 80, "B": 20, "C": 20, "D": 50}], index=[cur_date]),
        "macd_bull": pd.DataFrame([{"A": 0, "B": 1, "C": 1, "D": 1}], index=[cur_date]),
        "ma20": pd.DataFrame([{"A": hist["A"].iloc[-1] + 1, "B": 100, "C": 100, "D": 100}], index=[cur_date]),
        "ma60": pd.DataFrame([{"A": hist["A"].iloc[-1] + 1, "B": 100, "C": 100, "D": 100}], index=[cur_date]),
    }

    selected = select_robo_filtered_momentum(
        hist,
        base_strategy="quant",
        top_n=2,
        precomputed=precomputed,
        cur_date=cur_date,
    )

    assert selected == ["B", "C"]


def test_robo_filter_decision_reports_excluded_and_replacements() -> None:
    rows = 80
    index = pd.date_range("2024-01-01", periods=rows, freq="B")
    x = np.arange(rows)
    hist = pd.DataFrame(
        {
            "A": 100 + x * 1.00,
            "B": 100 + x * 0.70,
            "C": 100 + x * 0.50,
            "D": 100 + x * 0.20,
        },
        index=index,
    )
    cur_date = hist.index[-1]
    precomputed = {
        "rsi": pd.DataFrame([{"A": 80, "B": 20, "C": 20, "D": 50}], index=[cur_date]),
        "macd_bull": pd.DataFrame([{"A": 0, "B": 1, "C": 1, "D": 1}], index=[cur_date]),
        "ma20": pd.DataFrame([{"A": hist["A"].iloc[-1] + 1, "B": 100, "C": 100, "D": 100}], index=[cur_date]),
        "ma60": pd.DataFrame([{"A": hist["A"].iloc[-1] + 1, "B": 100, "C": 100, "D": 100}], index=[cur_date]),
    }

    decision = robo_filter_decision(
        hist,
        base_strategy="quant",
        top_n=2,
        precomputed=precomputed,
        cur_date=cur_date,
    )

    assert decision["raw_selected"] == ["A", "B"]
    assert decision["selected"] == ["B", "C"]
    assert decision["excluded"] == ["A"]
    assert decision["replacements"] == ["C"]
    assert decision["scores"]["A"] < 0


def test_select_for_backtest_dispatches_unknown_to_first_columns() -> None:
    assert select_for_backtest(_hist(80), "unknown", 2) == ["A", "B"]


def test_build_robo_precomputed_indicators_skips_short_series() -> None:
    index = pd.date_range("2024-01-02", periods=80, freq="B")
    price_df = pd.DataFrame(
        {
            "A": np.linspace(100, 130, 80),
            "B": [np.nan] * 30 + list(np.linspace(50, 60, 50)),
        },
        index=index,
    )

    result = build_robo_precomputed_indicators(price_df)

    assert set(result) == {"rsi", "macd_bull", "ma20", "ma60"}
    assert list(result["rsi"].columns) == ["A"]
    assert list(result["macd_bull"].columns) == ["A"]
    assert result["ma60"]["A"].notna().any()


def test_server_reuses_robo_precompute_helper() -> None:
    import server

    assert server._kq_build_robo_precomputed_indicators is build_robo_precomputed_indicators
    assert server._kq_select_for_backtest is select_for_backtest


def test_server_strategy_selector_uses_module_for_s2() -> None:
    import server

    assert server._select_for_bt(_momentum_split_hist(), "quant_s2", 1) == ["B"]
