"""API-facing risk-profile metadata for the recommendation tab.

Plan B deliberately does not change live recommendation weights. It wraps the
existing risk_profiles guardrail so the UI can show when an unvalidated preset
request was downgraded to the active neutral profile.
"""
from __future__ import annotations

from kq_tool.portfolio.risk_profiles import (
    RISK_PROFILES,
    ProfileStatus,
    get_risk_profile,
    profile_for_recommendation,
)

_STATUS_LABEL = {
    ProfileStatus.DEFINED: "정의됨 (검증 전)",
    ProfileStatus.ACTIVE: "활성",
    ProfileStatus.RETIRED: "비활성",
}


def _params_of(profile) -> dict[str, object]:
    """Return display-only preset parameters for the selected profile."""

    return {
        "regime_erc_n0": profile.regime_erc_n0,
        "band_abs": profile.rebalance_abs_band,
        "band_rel": profile.rebalance_rel_band,
        "regime_shift_ratio": profile.regime_shift_fraction,
        "asset_cap": profile.max_asset_cap,
        "promote_after": profile.promote_after,
        "validation_note": profile.validation_note,
    }


def _normalize_requested_key(requested_key: str | None) -> str:
    return str(requested_key or "neutral").strip().lower()


def resolve_profile_request(requested_key: str | None) -> dict[str, object]:
    """Return profile metadata to attach to recommendation API responses."""

    req_key = _normalize_requested_key(requested_key)
    requested = RISK_PROFILES.get(req_key)
    applied, _guardrail = profile_for_recommendation(req_key)

    if requested is None:
        return {
            "requested": req_key,
            "requested_label": "알 수 없음",
            "requested_status": None,
            "applied": applied.key,
            "applied_label": applied.label,
            "demoted": True,
            "notice": (
                f"'{req_key}'는 정의되지 않은 프리셋입니다. "
                f"기본값인 {applied.label} 기준으로 산출되었습니다."
            ),
            "params": None,
        }

    demoted = requested.key != applied.key
    if demoted:
        notice = (
            f"{requested.label}은 현재 '{_STATUS_LABEL.get(requested.status, requested.status.value)}' 단계입니다. "
            f"검증 게이트 통과 전 구성요소는 추천에 반영하지 않는 원칙에 따라 "
            f"{applied.label} 기준으로 산출되었습니다."
        )
    else:
        notice = f"{applied.label} 기준배분이 적용되었습니다."

    return {
        "requested": requested.key,
        "requested_label": requested.label,
        "requested_status": requested.status.value,
        "applied": applied.key,
        "applied_label": applied.label,
        "demoted": demoted,
        "notice": notice,
        "params": _params_of(requested),
    }


def list_profiles_for_ui() -> list[dict[str, object]]:
    """Return preset metadata for rendering selector pills."""

    items = []
    for key in ("stable", "neutral", "aggressive"):
        profile = get_risk_profile(key)
        items.append(
            {
                "key": profile.key,
                "label": profile.label,
                "status": profile.status.value,
                "active": profile.status == ProfileStatus.ACTIVE,
            }
        )
    return items

