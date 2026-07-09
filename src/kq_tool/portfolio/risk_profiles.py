"""Suitability-aware risk profile presets.

Risk profiles are a pre-registered *definition layer*. They are not wired into
live recommendations until their own validation gate passes. This mirrors the
project rule used elsewhere: define -> validate -> promote.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Mapping

from kq_tool.portfolio.rebalancing import RebalancePolicy
from kq_tool.portfolio.regime_erc import DEFAULT_REGIME_ASSET_BOUNDS, RegimeERCConfig


class ProfileStatus(StrEnum):
    DEFINED = "defined"
    ACTIVE = "active"
    RETIRED = "retired"


AssetBounds = dict[str, tuple[float, float]]
RegimeBounds = dict[str, AssetBounds]


CURRENT_ABS_BAND = 0.05
CURRENT_REL_BAND = 0.25
CURRENT_REGIME_SHIFT_FRACTION = 0.70
CURRENT_ASSET_CAP = 0.50


@dataclass(frozen=True)
class RiskProfile:
    """One investor-risk preset and its validation status."""

    key: str
    label: str
    score_range: tuple[int, int]
    description: str
    status: ProfileStatus
    regime_erc_n0: float
    rebalance_abs_band: float
    rebalance_rel_band: float
    regime_shift_fraction: float
    max_asset_cap: float
    regime_bounds: RegimeBounds
    rationale: Mapping[str, str]
    validation_note: str
    promote_after: str = "validate_risk_profiles.py PASS + review"
    audit_tags: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["status"] = self.status.value
        payload["regime_bounds"] = {
            regime: {asset: list(bounds) for asset, bounds in bounds_by_asset.items()}
            for regime, bounds_by_asset in self.regime_bounds.items()
        }
        return payload


STABLE_REGIME_BOUNDS: RegimeBounds = {
    "골디락스": {
        "stocks": (0.20, 0.55),
        "bonds": (0.20, 0.55),
        "gold": (0.03, 0.20),
        "cash": (0.10, 0.35),
    },
    "리플레이션": {
        "stocks": (0.20, 0.50),
        "bonds": (0.10, 0.45),
        "gold": (0.10, 0.35),
        "cash": (0.10, 0.40),
    },
    "스태그플레이션": {
        "stocks": (0.03, 0.20),
        "bonds": (0.15, 0.55),
        "gold": (0.15, 0.40),
        "cash": (0.25, 0.60),
    },
    "디플레이션": {
        "stocks": (0.03, 0.18),
        "bonds": (0.40, 0.75),
        "gold": (0.03, 0.20),
        "cash": (0.20, 0.60),
    },
}


AGGRESSIVE_REGIME_BOUNDS: RegimeBounds = {
    "골디락스": {
        "stocks": (0.45, 0.80),
        "bonds": (0.05, 0.30),
        "gold": (0.03, 0.15),
        "cash": (0.02, 0.20),
    },
    "리플레이션": {
        "stocks": (0.40, 0.75),
        "bonds": (0.03, 0.20),
        "gold": (0.10, 0.40),
        "cash": (0.02, 0.20),
    },
    "스태그플레이션": {
        "stocks": (0.10, 0.40),
        "bonds": (0.03, 0.30),
        "gold": (0.15, 0.45),
        "cash": (0.10, 0.45),
    },
    "디플레이션": {
        "stocks": (0.10, 0.35),
        "bonds": (0.25, 0.65),
        "gold": (0.05, 0.25),
        "cash": (0.10, 0.40),
    },
}


RISK_PROFILES: dict[str, RiskProfile] = {
    "stable": RiskProfile(
        key="stable",
        label="안정형",
        score_range=(0, 2),
        description="손실 회피를 우선해 채권·현금 허용폭을 넓히고 밴드를 좁게 관리",
        status=ProfileStatus.DEFINED,
        regime_erc_n0=60.0,
        rebalance_abs_band=0.03,
        rebalance_rel_band=0.20,
        regime_shift_fraction=0.50,
        max_asset_cap=0.70,
        regime_bounds=STABLE_REGIME_BOUNDS,
        rationale={
            "bond_cap": "[검증 후 확정] 변동성 연 5% 이내를 만족하는 채권/현금 허용폭으로 역산 예정",
            "band": "[검증 후 확정] 작은 손실 회피 성향을 반영해 중립형보다 좁은 밴드 후보",
            "status": "검증 전 정의 단계이므로 추천 요청 시 neutral로 강등",
        },
        validation_note="검증 러너에서 Vol<=5%, MDD<=10% 등 안정형 승격 기준을 통과해야 ACTIVE 가능",
        audit_tags=("suitability", "candidate", "downgrade_to_neutral"),
    ),
    "neutral": RiskProfile(
        key="neutral",
        label="중립형",
        score_range=(3, 4),
        description="현행 추천 탭 기준. ERC v1과 밴드 정책의 발표 데모 기본 설정",
        status=ProfileStatus.ACTIVE,
        regime_erc_n0=36.0,
        rebalance_abs_band=CURRENT_ABS_BAND,
        rebalance_rel_band=CURRENT_REL_BAND,
        regime_shift_fraction=CURRENT_REGIME_SHIFT_FRACTION,
        max_asset_cap=CURRENT_ASSET_CAP,
        regime_bounds={regime: dict(bounds) for regime, bounds in DEFAULT_REGIME_ASSET_BOUNDS.items()},
        rationale={
            "anchor": "현행 E2E 게이트를 통과한 발표 데모 기준값",
            "band": "절대 5%p·상대 25% 밴드",
            "regime_shift": "공식 국면 전환 시 70% 부분이동",
            "asset_cap": "ETF배분 위험기반 최적화 기본 자산별 50% 상한",
        },
        validation_note="현재 추천 탭에 연결된 유일한 ACTIVE 성향",
        audit_tags=("suitability", "active", "presentation_baseline"),
    ),
    "aggressive": RiskProfile(
        key="aggressive",
        label="공격형",
        score_range=(5, 6),
        description="주식 허용폭을 넓히고 밴드를 넓게 둬 장기 성장 노출을 유지",
        status=ProfileStatus.DEFINED,
        regime_erc_n0=24.0,
        rebalance_abs_band=0.07,
        rebalance_rel_band=0.35,
        regime_shift_fraction=0.80,
        max_asset_cap=0.80,
        regime_bounds=AGGRESSIVE_REGIME_BOUNDS,
        rationale={
            "stock_cap": "[검증 후 확정] 성장 노출 확대가 Sharpe/MDD 기준을 통과할 때만 승격",
            "band": "[검증 후 확정] 높은 회전율을 피하고 장기 보유를 허용하는 넓은 밴드 후보",
            "status": "검증 전 정의 단계이므로 추천 요청 시 neutral로 강등",
        },
        validation_note="검증 러너에서 Sharpe, MDD, 회전율 기준을 통과해야 ACTIVE 가능",
        audit_tags=("suitability", "candidate", "downgrade_to_neutral"),
    ),
}


def _normalize_profile_key(profile: object) -> str:
    key = str(profile or "neutral").strip().lower()
    aliases = {
        "defensive": "stable",
        "conservative": "stable",
        "safe": "stable",
        "안정형": "stable",
        "stable": "stable",
        "balanced": "neutral",
        "middle": "neutral",
        "neutral": "neutral",
        "중립형": "neutral",
        "growth": "aggressive",
        "aggressive": "aggressive",
        "공격형": "aggressive",
    }
    return aliases.get(key, key)


def get_risk_profile(profile: object = None) -> RiskProfile:
    """Return a risk profile definition, defaulting to neutral."""

    return RISK_PROFILES.get(_normalize_profile_key(profile), RISK_PROFILES["neutral"])


def profile_for_recommendation(profile: object = None) -> tuple[RiskProfile, dict[str, object]]:
    """Return the ACTIVE profile allowed for recommendation.

    Non-ACTIVE profiles are deliberately downgraded to neutral. This prevents
    unvalidated suitability presets from leaking into the live recommendation.
    """

    requested = get_risk_profile(profile)
    if requested.status == ProfileStatus.ACTIVE:
        return requested, {
            "requested_profile": requested.key,
            "active_profile": requested.key,
            "downgraded": False,
            "reason": "ACTIVE profile",
        }
    fallback = RISK_PROFILES["neutral"]
    return fallback, {
        "requested_profile": requested.key,
        "active_profile": fallback.key,
        "downgraded": True,
        "reason": f"{requested.key} is {requested.status.value}; validation required before promotion",
    }


def risk_profile_to_regime_erc_config(profile: object = None, *, for_recommendation: bool = True) -> RegimeERCConfig:
    """Build the ERC configuration for a profile.

    By default this uses the recommendation-safe profile, so unvalidated
    stable/aggressive requests collapse back to neutral.
    """

    preset = profile_for_recommendation(profile)[0] if for_recommendation else get_risk_profile(profile)
    return RegimeERCConfig(n0=preset.regime_erc_n0, regime_bounds=preset.regime_bounds)


def risk_profile_to_rebalance_policy(profile: object = None, *, for_recommendation: bool = True) -> RebalancePolicy:
    """Build the rebalancing policy for a profile."""

    preset = profile_for_recommendation(profile)[0] if for_recommendation else get_risk_profile(profile)
    return RebalancePolicy(
        abs_band=preset.rebalance_abs_band,
        rel_band=preset.rebalance_rel_band,
        regime_shift_fraction=preset.regime_shift_fraction,
    )


def classify_risk_profile(answers: Mapping[str, object] | None) -> dict[str, object]:
    """Classify three onboarding answers into a DEFINED/ACTIVE profile.

    Expected keys:
    - horizon_years
    - max_loss_pct
    - experience_level: none / medium / high or Korean equivalents
    """

    answers = answers or {}
    score = _horizon_score(answers.get("horizon_years"))
    score += _loss_score(answers.get("max_loss_pct"))
    score += _experience_score(answers.get("experience_level"))

    if score <= 2:
        preset = RISK_PROFILES["stable"]
    elif score <= 4:
        preset = RISK_PROFILES["neutral"]
    else:
        preset = RISK_PROFILES["aggressive"]

    active, guardrail = profile_for_recommendation(preset.key)
    return {
        "score": score,
        "profile": preset.key,
        "label": preset.label,
        "status": preset.status.value,
        "active_profile": active.key,
        "active_label": active.label,
        "downgraded_for_recommendation": guardrail["downgraded"],
        "reason": guardrail["reason"],
    }


def risk_profile_table() -> list[dict[str, object]]:
    """Return all profile definitions for UI/reporting."""

    return [RISK_PROFILES[key].to_dict() for key in ("stable", "neutral", "aggressive")]


def active_profile_keys() -> tuple[str, ...]:
    """Return profiles currently allowed in live recommendation."""

    return tuple(key for key, profile in RISK_PROFILES.items() if profile.status == ProfileStatus.ACTIVE)


def _horizon_score(years: object) -> int:
    try:
        value = float(years)
    except Exception:
        return 1
    if value < 2:
        return 0
    if value < 5:
        return 1
    return 2


def _loss_score(loss_pct: object) -> int:
    try:
        value = float(loss_pct)
    except Exception:
        return 1
    if value < 10:
        return 0
    if value < 25:
        return 1
    return 2


def _experience_score(level: object) -> int:
    text = str(level or "").strip().lower()
    if text in {"none", "beginner", "초보", "없음", "무경험"}:
        return 0
    if text in {"high", "expert", "experienced", "많음", "전문"}:
        return 2
    return 1
