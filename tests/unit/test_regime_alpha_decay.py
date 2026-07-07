from __future__ import annotations

import pandas as pd

from kq_tool.validation.regime_alpha_decay import (
    random_return_pool_by_regime,
    regime_at_date,
    regime_summary_rows,
    split_events_by_regime,
    summarize_regime_alpha_decay,
)


def _price_frame() -> pd.DataFrame:
    index = pd.date_range("2021-01-01", periods=8, freq="D")
    close = pd.Series([100.0, 101.0, 103.0, 102.0, 104.0, 107.0, 109.0, 110.0], index=index)
    return pd.DataFrame({"Close": close}, index=index)


def test_regime_at_date_uses_latest_known_value() -> None:
    regimes = pd.Series(
        ["A", "B"],
        index=pd.to_datetime(["2021-01-03", "2021-01-06"]),
    )

    assert regime_at_date(regimes, pd.Timestamp("2021-01-02"), fallback="X") == "X"
    assert regime_at_date(regimes, pd.Timestamp("2021-01-05"), fallback="X") == "A"
    assert regime_at_date(regimes, pd.Timestamp("2021-01-08"), fallback="X") == "B"


def test_split_events_by_regime_groups_by_event_date() -> None:
    regimes = pd.Series(
        ["A", "B"],
        index=pd.to_datetime(["2021-01-01", "2021-01-06"]),
    )
    events = {
        "buy": [
            ("AAA", pd.Timestamp("2021-01-04"), 1, 1, 0.01),
            ("BBB", pd.Timestamp("2021-01-07"), 1, 1, 0.02),
        ]
    }

    grouped = split_events_by_regime(events, regimes)

    assert grouped["A"]["buy"][0][0] == "AAA"
    assert grouped["B"]["buy"][0][0] == "BBB"


def test_random_return_pool_by_regime_filters_candidate_dates() -> None:
    frame = _price_frame()
    regimes = pd.Series(
        ["A", "A", "A", "A", "B", "B", "B", "B"],
        index=frame.index,
    )

    pool_a = random_return_pool_by_regime(
        {"AAA": frame},
        regimes,
        target_regime="A",
        horizon=1,
        direction=1,
        limit_threshold=0.30,
        oos_start=pd.Timestamp("2021-01-01"),
    )
    pool_b = random_return_pool_by_regime(
        {"AAA": frame},
        regimes,
        target_regime="B",
        horizon=1,
        direction=1,
        limit_threshold=0.30,
        oos_start=pd.Timestamp("2021-01-01"),
    )

    assert len(pool_a) == 4
    assert len(pool_b) == 3
    assert pool_a.mean() != pool_b.mean()


def test_summarize_regime_alpha_decay_compares_actual_to_same_regime_random_pool() -> None:
    frame = _price_frame()
    regimes = pd.Series(["A"] * len(frame), index=frame.index)
    events = {
        "buy": [
            ("AAA", pd.Timestamp("2021-01-02"), 1, 1, 0.02),
            ("AAA", pd.Timestamp("2021-01-03"), 1, 1, 0.03),
        ]
    }

    summary = summarize_regime_alpha_decay(
        events,
        {"AAA": frame},
        regimes,
        limit_threshold=0.30,
        n=20,
        seed=7,
        oos_start=pd.Timestamp("2021-01-01"),
        regimes=["A"],
    )
    rows = regime_summary_rows(summary)

    assert summary["A"]["n_events"] == 2
    assert summary["A"]["actual_mean"] == 0.025
    assert summary["A"]["random_mean"] is not None
    assert 0.0 <= summary["A"]["p_value"] <= 1.0
    assert rows[0]["regime"] == "A"
    assert rows[0]["events"] == 2
