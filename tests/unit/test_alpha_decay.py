from __future__ import annotations

import numpy as np
import pandas as pd

from kq_tool.analyzer.alpha_decay import (
    DECAY_FAMILY_PRIORS,
    alpha_single,
    classify_decay_quality,
    half_life_to_confidence,
)


def _sample_close() -> pd.Series:
    index = pd.date_range("2022-01-03", periods=180, freq="B")
    base = np.linspace(100, 125, len(index))
    cycle = np.sin(np.arange(len(index)) / 3) * 6
    return pd.Series(base + cycle, index=index, name="Close")


def test_half_life_to_confidence_is_conservative_for_missing_values() -> None:
    assert half_life_to_confidence(None, 30) == 2.0
    assert half_life_to_confidence(7, 30) > half_life_to_confidence(None, 30)


def test_half_life_to_confidence_discounts_small_sample_count() -> None:
    assert half_life_to_confidence(7, 5) < half_life_to_confidence(7, 30)


def test_alpha_single_returns_signal_metadata() -> None:
    result = alpha_single(_sample_close(), horizons=[1, 3, 5, 7], active_window=5)

    assert result
    for detail in result.values():
        assert "count" in detail
        assert "raw_count" in detail
        assert "excluded_limit" in detail
        assert "active_window" in detail
        assert "direction" in detail
        assert detail["decay_state"] in {"measured", "imputed", "neutral"}
        assert "decay_reason_code" in detail
        assert detail["active_window"] == 5


def test_decay_quality_marks_stable_large_sample_as_measured() -> None:
    result = classify_decay_quality(
        "RSI 과매도", {"count": 40, "half_life": 7.0, "is_active": True}
    )

    assert result["decay_state"] == "measured"
    assert result["effective_half_life"] == 7.0
    assert result["decay_basis"] == "half_life"


def test_decay_quality_imputes_family_prior_for_small_active_sample() -> None:
    result = classify_decay_quality(
        "RSI 과매도", {"count": 4, "half_life": None, "is_active": True}
    )

    assert result["decay_state"] == "imputed"
    assert result["decay_basis"] == "family_prior"
    assert result["effective_half_life"] == DECAY_FAMILY_PRIORS["RSI 과매도"]["half_life"]


def test_decay_quality_neutralizes_unstable_large_sample() -> None:
    result = classify_decay_quality(
        "RSI 과매도", {"count": 40, "half_life": None, "is_active": True}
    )

    assert result["decay_state"] == "neutral"
    assert result["effective_half_life"] is None
    assert result["decay_basis"] == "neutral"


def test_decay_family_priors_do_not_depend_on_imputed_signal_results() -> None:
    for prior in DECAY_FAMILY_PRIORS.values():
        assert set(prior) == {"family", "half_life"}
        assert prior["half_life"] > 0


def test_alpha_single_defaults_to_operating_active_window() -> None:
    result = alpha_single(_sample_close(), horizons=[1, 3, 5, 7])

    assert {detail["active_window"] for detail in result.values()} == {10}


def test_alpha_single_can_use_custom_active_window() -> None:
    result = alpha_single(_sample_close(), horizons=[1, 3, 5, 7], active_window=10)

    assert {detail["active_window"] for detail in result.values()} == {10}


def test_alpha_single_can_filter_events_by_signal_quality() -> None:
    close = _sample_close()
    signal_names = [
        "RSI 과매도",
        "RSI 과매수",
        "MACD 골든크로스",
        "MACD 데드크로스",
        "BB 하단터치",
        "BB 상단터치",
    ]
    quality = pd.DataFrame(0.0, index=close.index, columns=signal_names)
    quality.loc[close.index[-8:], "RSI 과매도"] = 0.9

    result = alpha_single(
        close,
        horizons=[1, 3, 5, 7],
        quality_scores=quality,
        min_signal_quality=0.6,
        weight_by_quality=True,
    )
    detail = result["RSI 과매도"]

    assert detail["quality_count"] == 8
    assert detail["count"] == 8
    assert detail["avg_quality"] == 0.9
    assert detail["quality_threshold"] == 0.6
    assert detail["quality_weighted"] is True
    assert detail["is_active"] is True
