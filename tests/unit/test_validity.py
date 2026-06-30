from __future__ import annotations

from kq_tool.portfolio.validity import portfolio_validity


def test_portfolio_validity_direct_module_import() -> None:
    result = portfolio_validity(
        {
            "current": "디플레이션",
            "confidence": 0.6,
            "duration_avg": {"디플레이션": 0.2},
            "next_quarter": {"디플레이션": 0.4},
        }
    )

    assert 10 <= result["valid_days"] <= 90
    assert result["regime_duration_quarters"] == 0.2
    assert result["next_same_regime_prob"] == 40.0


def test_portfolio_validity_uses_minimum_review_window_for_short_regimes() -> None:
    result = portfolio_validity(
        {
            "current": "골디락스",
            "confidence": 0.0,
            "duration_avg": {"골디락스": 0.01},
            "next_quarter": {},
        }
    )

    assert result["valid_days"] == 10
    assert result["valid_weeks"] == 2.0


def test_portfolio_validity_caps_long_high_confidence_review_window() -> None:
    result = portfolio_validity(
        {
            "current": "리플레이션",
            "confidence": 1.0,
            "duration_avg": {"리플레이션": 4.0},
            "next_quarter": {"리플레이션": 0.75},
        }
    )

    assert result["valid_days"] == 90
    assert result["next_same_regime_prob"] == 75.0


def test_portfolio_validity_expands_window_with_confidence() -> None:
    low = portfolio_validity(
        {
            "current": "스태그플레이션",
            "confidence": 0.2,
            "duration_avg": {"스태그플레이션": 1.0},
        }
    )
    high = portfolio_validity(
        {
            "current": "스태그플레이션",
            "confidence": 0.9,
            "duration_avg": {"스태그플레이션": 1.0},
        }
    )

    assert high["valid_days"] > low["valid_days"]
