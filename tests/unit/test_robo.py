from __future__ import annotations

import kq_tool.analyzer.robo as robo
from kq_tool.analyzer.robo import confidence_weighted_robo, legacy_robo_score, score_to_signal
from kq_tool.config import ROBO_SIGNAL_LABELS, ROBO_SIGNAL_WEIGHTS, SIGNAL_DIRECTION
from kq_tool.config import ROBO_BUY_THRESHOLD, ROBO_SELL_THRESHOLD


def test_score_to_signal_thresholds() -> None:
    assert score_to_signal(ROBO_BUY_THRESHOLD) == "매수"
    assert score_to_signal(ROBO_SELL_THRESHOLD) == "매도"
    assert score_to_signal(50) == "관망"


def test_legacy_robo_score_matches_original_scale() -> None:
    assert legacy_robo_score(1, 1, 1, 1, 1) == 100
    assert legacy_robo_score(-1, -1, -1, -1, -1) == 0
    assert legacy_robo_score(0, 0, 0, 0, 0) == 50


def test_confidence_weighted_robo_uses_alpha_decay_validity() -> None:
    alpha = {
        "RSI 과매도": {
            "is_active": True,
            "half_life": 6,
            "count": 30,
            "horizon_rets": {"1": 0.1},
        }
    }

    result = confidence_weighted_robo(alpha, 1, 0, 0, 0, 0)

    assert result["validity_basis"] == "alpha_decay"
    assert result["exit_days"] == 6
    assert result["detail"][0]["signal_name"] == "RSI 과매도"


def test_confidence_weighted_robo_falls_back_to_review_interval() -> None:
    result = confidence_weighted_robo({}, 0, 0, 0, 0, 0)

    assert result["signal"] == "관망"
    assert result["validity_basis"] == "review_interval"
    assert result["valid_days"] == 10


def test_robo_uses_config_signal_direction() -> None:
    assert robo.SIGNAL_DIRECTION == SIGNAL_DIRECTION


def test_robo_uses_config_weights_and_labels() -> None:
    assert robo.BASE_WEIGHT == ROBO_SIGNAL_WEIGHTS
    assert robo.SIGNAL_LABELS == ROBO_SIGNAL_LABELS
