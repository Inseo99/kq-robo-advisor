from __future__ import annotations

import pandas as pd

from kq_tool.data.marketcap import (
    MCAP_KEY,
    build_mcap_history,
    build_top_marketcap_tickers,
    get_top_mcap_at,
)
from kq_tool.utils.tickers import ticker_to_code


def _fin_data() -> pd.DataFrame:
    index = pd.MultiIndex.from_tuples(
        [
            ("000001", MCAP_KEY),
            ("000001", MCAP_KEY),
            ("000002", MCAP_KEY),
            ("000002", MCAP_KEY),
            ("000003", MCAP_KEY),
            ("000003", MCAP_KEY),
        ]
    )
    return pd.DataFrame(
        {
            "date": [
                "2020-12-31",
                "2021-03-31",
                "2020-12-31",
                "2021-03-31",
                "2020-12-31",
                "2021-03-31",
            ],
            "value": [100, 90, 80, 120, 60, 70],
        },
        index=index,
    )


def test_build_top_marketcap_tickers_uses_latest_values() -> None:
    universe = {"000001.KS": ("A",), "000002.KS": ("B",), "000003.KS": ("C",)}

    result = build_top_marketcap_tickers(universe, ticker_to_code, _fin_data(), limit=2)

    assert result == ["000002.KS", "000001.KS"]


def test_build_mcap_history_returns_forward_filled_frame() -> None:
    universe = {"000001.KS": ("A",), "000002.KS": ("B",), "000003.KS": ("C",)}

    history = build_mcap_history(universe, ticker_to_code, _fin_data())

    assert list(history.columns) == ["000001.KS", "000002.KS", "000003.KS"]
    assert history.loc[pd.Timestamp("2021-05-15"), "000002.KS"] == 120


def test_get_top_mcap_at_is_point_in_time() -> None:
    universe = {"000001.KS": ("A",), "000002.KS": ("B",), "000003.KS": ("C",)}
    history = build_mcap_history(universe, ticker_to_code, _fin_data())

    assert get_top_mcap_at(history, "2021-04-15", n=2) == ["000001.KS", "000002.KS"]
    assert get_top_mcap_at(history, "2021-05-20", n=2) == ["000002.KS", "000001.KS"]


def test_get_top_mcap_at_returns_empty_before_first_known_date() -> None:
    universe = {"000001.KS": ("A",)}
    history = build_mcap_history(universe, ticker_to_code, _fin_data())

    assert get_top_mcap_at(history, "2019-12-31", n=2) == []
