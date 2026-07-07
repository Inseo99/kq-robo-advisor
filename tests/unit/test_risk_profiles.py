from dataclasses import FrozenInstanceError

import pytest

from kq_tool.portfolio.risk_profiles import (
    CURRENT_ABS_BAND,
    CURRENT_ASSET_CAP,
    CURRENT_REGIME_SHIFT_FRACTION,
    CURRENT_REL_BAND,
    RISK_PROFILES,
    ProfileStatus,
    active_profile_keys,
    classify_risk_profile,
    get_risk_profile,
    profile_for_recommendation,
    risk_profile_table,
    risk_profile_to_rebalance_policy,
)


def test_only_neutral_is_active():
    assert active_profile_keys() == ("neutral",)
    assert RISK_PROFILES["neutral"].status == ProfileStatus.ACTIVE
    assert RISK_PROFILES["stable"].status == ProfileStatus.DEFINED
    assert RISK_PROFILES["aggressive"].status == ProfileStatus.DEFINED


def test_neutral_matches_current_defaults():
    neutral = RISK_PROFILES["neutral"]
    assert neutral.rebalance_abs_band == CURRENT_ABS_BAND == 0.05
    assert neutral.rebalance_rel_band == CURRENT_REL_BAND == 0.25
    assert neutral.regime_shift_fraction == CURRENT_REGIME_SHIFT_FRACTION == 0.70
    assert neutral.max_asset_cap == CURRENT_ASSET_CAP == 0.50

    policy = risk_profile_to_rebalance_policy("neutral")
    assert policy.abs_band == 0.05
    assert policy.rel_band == 0.25
    assert policy.regime_shift_fraction == 0.70


@pytest.mark.parametrize("requested", ["stable", "aggressive", "안정형", "공격형"])
def test_unvalidated_profiles_downgrade_to_neutral(requested):
    active, audit = profile_for_recommendation(requested)
    assert active.key == "neutral"
    assert audit["downgraded"] is True
    assert audit["active_profile"] == "neutral"


@pytest.mark.parametrize(
    ("answers", "expected_profile", "expected_active"),
    [
        ({"horizon_years": 1, "max_loss_pct": 5, "experience_level": "none"}, "stable", "neutral"),
        ({"horizon_years": 3, "max_loss_pct": 15, "experience_level": "medium"}, "neutral", "neutral"),
        ({"horizon_years": 8, "max_loss_pct": 35, "experience_level": "high"}, "aggressive", "neutral"),
    ],
)
def test_three_question_onboarding_mapping(answers, expected_profile, expected_active):
    result = classify_risk_profile(answers)
    assert result["profile"] == expected_profile
    assert result["active_profile"] == expected_active


def test_risk_profile_dataclass_is_frozen():
    profile = get_risk_profile("neutral")
    with pytest.raises(FrozenInstanceError):
        profile.status = ProfileStatus.DEFINED  # type: ignore[misc]


def test_rationale_is_required_for_all_profiles():
    rows = risk_profile_table()
    assert {row["key"] for row in rows} == {"stable", "neutral", "aggressive"}
    for row in rows:
        assert row["rationale"]
        assert row["validation_note"]
        if row["key"] != "neutral":
            assert "검증" in str(row["validation_note"])
