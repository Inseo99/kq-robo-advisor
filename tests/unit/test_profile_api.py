"""Contracts for the recommendation risk-profile API layer."""

import pytest

from kq_tool.portfolio.profile_api import list_profiles_for_ui, resolve_profile_request


def test_neutral_request_not_demoted():
    result = resolve_profile_request("neutral")
    assert result["demoted"] is False
    assert result["applied"] == "neutral"
    assert "중립형" in result["notice"]


@pytest.mark.parametrize("key,label", [("stable", "안정형"), ("aggressive", "공격형")])
def test_unvalidated_request_demoted_with_reason(key, label):
    result = resolve_profile_request(key)
    assert result["requested"] == key
    assert result["applied"] == "neutral"
    assert result["demoted"] is True
    assert label in result["notice"]
    assert "검증" in result["notice"] and "원칙" in result["notice"]
    assert result["requested_status"] == "defined"


@pytest.mark.parametrize("key", [None, "", "unknown_key", "STABLE  "])
def test_edge_inputs_are_safe(key):
    result = resolve_profile_request(key)
    assert result["applied"] == "neutral"
    assert isinstance(result["notice"], str) and result["notice"]


def test_stable_uppercase_is_normalized():
    result = resolve_profile_request("  Stable ")
    assert result["requested"] == "stable"
    assert result["demoted"] is True


def test_ui_list_shape():
    items = list_profiles_for_ui()
    assert {item["key"] for item in items} == {"stable", "neutral", "aggressive"}
    assert [item["key"] for item in items if item["active"]] == ["neutral"]
