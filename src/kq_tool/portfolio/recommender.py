"""Recommendation portfolio helper rules."""

from __future__ import annotations

import csv
from collections.abc import Mapping, Sequence
from pathlib import Path

from kq_tool.portfolio.validity import portfolio_validity
from kq_tool.portfolio.transition_allocation import (
    TEAConfig,
    transition_expected_regime_target,
)
from kq_tool.portfolio.weights import combine_weight_sets, normalize_weights

META_COMPONENTS = [
    {"name": "동일비중", "weight": 0.40, "reason": "OOS 최저 MDD/최고 Calmar 후보"},
    {"name": "영구포트폴리오", "weight": 0.35, "reason": "OOS 최고 Sharpe 후보"},
    {"name": "올웨더", "weight": 0.25, "reason": "채권 중심 방어 배분"},
]

REGIME_TARGETS = {
    "골디락스": {
        "069500.KS": 0.45,
        "229200.KS": 0.15,
        "148070.KS": 0.20,
        "132030.KS": 0.05,
        "153130.KS": 0.10,
        "114260.KS": 0.05,
    },
    "리플레이션": {
        "069500.KS": 0.32,
        "229200.KS": 0.08,
        "130680.KS": 0.18,
        "132030.KS": 0.22,
        "114260.KS": 0.08,
        "153130.KS": 0.12,
    },
    "스태그플레이션": {
        "069500.KS": 0.15,
        "229200.KS": 0.03,
        "130680.KS": 0.12,
        "132030.KS": 0.35,
        "114260.KS": 0.10,
        "153130.KS": 0.25,
    },
    "디플레이션": {
        "069500.KS": 0.18,
        "229200.KS": 0.02,
        "148070.KS": 0.45,
        "114260.KS": 0.15,
        "153130.KS": 0.15,
        "132030.KS": 0.05,
    },
}

REGIME_ALPHA_MIN_EVENTS = 50


def _to_float(value: object) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except Exception:
        return None


def _to_int(value: object) -> int:
    parsed = _to_float(value)
    return int(parsed) if parsed is not None else 0


def load_regime_alpha_summary(path: str | Path | None) -> list[dict[str, object]]:
    """Load a regime Alpha Decay summary CSV if it exists.

    The recommendation layer treats this as optional evidence. Missing or
    malformed files return an empty list so the portfolio stays neutral.
    """

    if not path:
        return []
    try:
        csv_path = Path(path)
        if not csv_path.exists() or not csv_path.is_file():
            return []
        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    except Exception:
        return []


def regime_alpha_signal_adjustment(
    current_regime: object,
    summary_rows: Sequence[Mapping[str, object]] | None,
    *,
    min_events: int = REGIME_ALPHA_MIN_EVENTS,
) -> dict[str, object]:
    """Convert regime Alpha Decay validation evidence into a tilt scale.

    The returned scale changes only the *strength* of robo/Alpha Decay weight
    tilts. It never flips the underlying buy/hold/sell signal.
    """

    regime = str(current_regime or "")
    base = {
        "regime": regime or None,
        "status": "자료없음",
        "signal_tilt_scale": 1.0,
        "events": 0,
        "placebo_n": None,
        "edge_pct": None,
        "random_pct": None,
        "excess_pct": None,
        "p_value": None,
        "rule": "국면별 Alpha Decay 검증 파일이 없으면 기존 로보/Alpha Decay 미세조정을 그대로 사용",
    }
    if not regime or not summary_rows:
        return base

    selected = None
    for row in summary_rows:
        if str(row.get("regime") or "") == regime:
            selected = row
            break
    if selected is None:
        return base

    events = _to_int(selected.get("events"))
    placebo_n = _to_int(selected.get("n") or selected.get("placebo_n"))
    edge = _to_float(selected.get("edge_pct"))
    random = _to_float(selected.get("random_pct"))
    excess = _to_float(selected.get("excess_pct"))
    p_value = _to_float(selected.get("p_value"))

    status = "중립"
    scale = 1.0
    rule = "같은 국면 내 무작위 날짜 대비 우위가 작아 기존 신호 강도 유지"
    if placebo_n <= 0:
        status = "검증횟수없음"
        scale = 1.0
        rule = "placebo 반복 수가 기록되지 않은 summary는 운영 신호 조정에 사용하지 않음"
    elif placebo_n < 50:
        status = "검증부족"
        scale = 1.0
        rule = f"placebo 반복 수 {placebo_n}회로 부족해 운영 신호 강도는 기존대로 유지"
    elif events < int(min_events):
        status = "표본부족"
        scale = 0.85
        rule = f"국면별 OOS 이벤트 {events}개로 표본이 부족해 신호 미세조정을 보수적으로 약화"
    elif excess is None:
        status = "중립"
        scale = 1.0
        rule = "국면별 excess 산출값이 없어 기존 신호 강도 유지"
    elif excess <= -0.25:
        status = "약화"
        scale = 0.55
        rule = "같은 국면 내 무작위 날짜보다 실제 신호 성과가 낮아 로보/Alpha Decay 틸트 강도 축소"
    elif excess < 0:
        status = "소폭약화"
        scale = 0.75
        rule = "같은 국면 내 초과 edge가 음수라 신호 틸트 강도를 소폭 축소"
    elif excess >= 0.75 and (p_value is None or p_value <= 0.25):
        status = "강화"
        scale = 1.10
        rule = "같은 국면 내 실제 신호 edge가 무작위 대비 충분히 높아 신호 틸트 강도 소폭 강화"
    elif excess >= 0.25:
        status = "소폭강화"
        scale = 1.05
        rule = "같은 국면 내 실제 신호 edge가 무작위보다 높아 신호 틸트 강도 소폭 강화"

    return {
        "regime": regime,
        "status": status,
        "signal_tilt_scale": round(float(max(0.0, min(1.15, scale))), 3),
        "events": events,
        "placebo_n": placebo_n,
        "edge_pct": None if edge is None else round(edge, 3),
        "random_pct": None if random is None else round(random, 3),
        "excess_pct": None if excess is None else round(excess, 3),
        "p_value": None if p_value is None else round(p_value, 4),
        "rule": rule,
    }


def regime_probability_blend(
    snapshot: Mapping[str, object],
    *,
    current_weight: float = 0.70,
    next_weight: float = 0.30,
    regimes: list[str] | tuple[str, ...] | None = None,
) -> dict[str, float]:
    """Blend current and next-quarter regime probabilities into one PiT mix."""

    regime_names = list(regimes or REGIME_TARGETS.keys())
    current_probs = snapshot.get("probs") or {}
    if not isinstance(current_probs, Mapping):
        current_probs = {}
    next_probs = snapshot.get("next_quarter") or {}
    if not isinstance(next_probs, Mapping):
        next_probs = {}

    current = snapshot.get("current") or snapshot.get("current_regime")
    if not current_probs and current:
        current_probs = {str(current): 1.0}

    blended: dict[str, float] = {}
    for regime in regime_names:
        try:
            cur_prob = float(current_probs.get(regime, 0.0))
        except Exception:
            cur_prob = 0.0
        try:
            nxt_prob = float(next_probs.get(regime, 0.0))
        except Exception:
            nxt_prob = 0.0
        prob = current_weight * cur_prob + next_weight * nxt_prob
        if prob > 0:
            blended[regime] = prob

    if not blended and current in regime_names:
        blended[str(current)] = 1.0
    return normalize_weights(blended)


def probability_weighted_regime_target(
    snapshot: Mapping[str, object],
    regime_targets: Mapping[str, Mapping[str, float]] | None = None,
    *,
    current_weight: float = 0.70,
    next_weight: float = 0.30,
) -> dict[str, float]:
    """Build an asset target from all regime probabilities instead of one hard label."""

    targets = regime_targets or REGIME_TARGETS
    if snapshot.get("transition_matrix"):
        try:
            result = transition_expected_regime_target(
                snapshot,
                targets,
                TEAConfig(horizon=3, p_stay_lo=0.50, p_stay_hi=0.80),
            )
            return normalize_weights(result.weights.to_dict())
        except Exception:
            pass

    blend = regime_probability_blend(
        snapshot,
        current_weight=current_weight,
        next_weight=next_weight,
        regimes=tuple(targets.keys()),
    )
    if not blend:
        current = snapshot.get("current") or snapshot.get("current_regime") or "리플레이션"
        fallback = targets.get(str(current)) or targets.get("리플레이션") or {}
        return normalize_weights(fallback)

    return combine_weight_sets(
        (probability, targets.get(regime))
        for regime, probability in blend.items()
        if targets.get(regime)
    )


def auto_regime_tilt(snapshot: Mapping[str, object]) -> dict:
    """Return conservative regime-tilt strength from confidence and stability."""

    confidence = float(snapshot.get("confidence", 0.0) or 0.0)
    current = snapshot.get("current")
    next_quarter = snapshot.get("next_quarter") or {}

    stay_prob = None
    if current in next_quarter:  # type: ignore[operator]
        try:
            stay_prob = float(next_quarter[current])  # type: ignore[index]
        except Exception:
            stay_prob = None

    if confidence < 0.50:
        return {
            "regime_tilt": 0.0,
            "confidence_level": "무틸트",
            "confidence": round(confidence * 100, 1),
            "next_same_regime_prob": round(stay_prob * 100, 1) if stay_prob is not None else None,
            "rule": "신뢰도 50% 미만은 베이스 포트폴리오 그대로 유지 (검증 결과 기반)",
        }
    if confidence < 0.70:
        regime_tilt = 0.08
        level = "낮음"
    elif confidence < 0.85:
        regime_tilt = 0.15
        level = "보통"
    else:
        regime_tilt = 0.20
        level = "높음"

    if stay_prob is not None:
        if stay_prob < 0.35:
            regime_tilt *= 0.50
            stability_note = "국면 전환 임박, 틸트 약화"
        elif stay_prob > 0.60:
            stability_note = "국면 안정 유지"
        else:
            stability_note = "국면 중립"
    else:
        stability_note = "다음분기 정보 없음"

    regime_tilt = float(max(0.0, min(0.22, regime_tilt)))
    return {
        "regime_tilt": regime_tilt,
        "confidence_level": level,
        "confidence": round(confidence * 100, 1),
        "next_same_regime_prob": round(stay_prob * 100, 1) if stay_prob is not None else None,
        "stability_note": stability_note,
        "rule": "신뢰도 70%+ 보통 틸트, 85%+ 강한 틸트, 50% 미만은 베이스 유지 (검증 기반)",
    }


def signal_weight_multiplier(signal: Mapping[str, object], *, signal_tilt_scale: float = 1.0) -> float:
    """Convert robo/Alpha-Decay signal into a small portfolio weight multiplier."""

    action = signal.get("cw_signal") or signal.get("signal") or "관망"
    try:
        confidence = float(signal.get("confidence") or 0.0)
    except Exception:
        confidence = 0.0
    if confidence > 1.0:
        confidence = confidence / 10.0
    confidence = max(0.0, min(1.0, confidence))

    try:
        exit_days = signal.get("exit_days")
        exit_days = float(exit_days) if exit_days is not None else None
    except Exception:
        exit_days = None

    decay_factor = 1.0
    if exit_days is not None:
        if exit_days <= 3:
            decay_factor = 0.35
        elif exit_days <= 7:
            decay_factor = 0.70
        elif exit_days >= 20:
            decay_factor = 0.85

    strength = confidence * decay_factor
    if action == "매수":
        base_multiplier = 1.0 + 0.10 * strength
    elif action == "매도":
        base_multiplier = 1.0 - 0.12 * strength
    else:
        base_multiplier = 1.0

    try:
        scale = float(signal_tilt_scale)
    except Exception:
        scale = 1.0
    scale = max(0.0, min(1.20, scale))
    multiplier = 1.0 + (base_multiplier - 1.0) * scale

    return round(float(max(0.75, min(1.12, multiplier))), 4)


def apply_signal_tilt(
    weights: Mapping[str, float],
    signal_map: Mapping[str, Mapping[str, object]],
    *,
    signal_tilt_scale: float = 1.0,
) -> tuple[dict[str, float], dict[str, float]]:
    """Apply robo signal multipliers and renormalize portfolio weights."""

    adjusted = {}
    multipliers = {}
    for ticker, weight in weights.items():
        multiplier = signal_weight_multiplier(signal_map.get(ticker, {}), signal_tilt_scale=signal_tilt_scale)
        multipliers[ticker] = multiplier
        adjusted[ticker] = float(weight) * multiplier
    return normalize_weights(adjusted), multipliers


def default_asset_signal(error: str | None = None) -> dict:
    """Return the neutral recommendation signal used when analysis is unavailable."""

    return {
        "signal": "관망",
        "score": None,
        "cw_signal": "관망",
        "cw_score": None,
        "confidence": None,
        "exit_days": None,
        "cur": None,
        "chg": None,
        "cur_date": None,
        "error": error,
    }


def stock_analysis_to_asset_signal(analysis: Mapping[str, object]) -> dict:
    """Summarize a stock-analysis payload into recommendation signal fields."""

    signal = default_asset_signal()
    robo = analysis.get("robo", {}) if isinstance(analysis, Mapping) else {}
    if not isinstance(robo, Mapping):
        robo = {}
    signal.update(
        {
            "signal": robo.get("signal"),
            "score": robo.get("score"),
            "cw_signal": robo.get("cw_signal"),
            "cw_score": robo.get("cw_score"),
            "confidence": robo.get("confidence"),
            "exit_days": robo.get("exit_days"),
            "cur": analysis.get("cur"),
            "chg": analysis.get("chg"),
            "cur_date": analysis.get("cur_date"),
        }
    )
    return signal


def build_recommendation_report(
    *,
    snapshot: Mapping[str, object],
    base_weights: Mapping[str, float],
    regime_target: Mapping[str, float] | None = None,
    signal_map: Mapping[str, Mapping[str, object]],
    etf_meta: Mapping[str, tuple[str, str, str]],
    components: list[Mapping[str, object]] | None = None,
    regime_alpha_summary: Sequence[Mapping[str, object]] | None = None,
) -> dict:
    """Build the complete recommendation portfolio payload."""

    auto = auto_regime_tilt(snapshot)
    regime_tilt = float(auto["regime_tilt"])
    regime_blend = regime_probability_blend(snapshot)
    tea_payload = None
    if snapshot.get("transition_matrix"):
        try:
            tea_result = transition_expected_regime_target(
                snapshot,
                REGIME_TARGETS,
                TEAConfig(horizon=3, p_stay_lo=0.50, p_stay_hi=0.80),
            )
            tea_payload = tea_result.to_payload()
        except Exception:
            tea_payload = None
    if regime_target is None:
        if tea_payload:
            regime_target = tea_payload["weights"]
        else:
            regime_target = probability_weighted_regime_target(snapshot)
    else:
        regime_target = normalize_weights(regime_target)
    pre_signal_weights = combine_weight_sets(
        [(1.0 - regime_tilt, base_weights), (regime_tilt, regime_target)]
    )
    validity = portfolio_validity(snapshot)
    current = snapshot.get("current") or snapshot.get("current_regime")
    regime_alpha = regime_alpha_signal_adjustment(current, regime_alpha_summary)
    final_weights, signal_multipliers = apply_signal_tilt(
        pre_signal_weights,
        signal_map,
        signal_tilt_scale=float(regime_alpha.get("signal_tilt_scale", 1.0) or 1.0),
    )

    assets = []
    for ticker, weight in sorted(final_weights.items(), key=lambda item: item[1], reverse=True):
        if weight < 0.005:
            continue
        name, asset_type, stance = etf_meta.get(ticker, (ticker, "기타", ""))
        assets.append(
            {
                "ticker": ticker,
                "name": name,
                "asset_type": asset_type,
                "stance": stance,
                "weight": round(weight * 100, 1),
                "pre_signal_weight": round(pre_signal_weights.get(ticker, 0.0) * 100, 1),
                "signal_multiplier": signal_multipliers.get(ticker, 1.0),
                "signal": signal_map.get(ticker, {}),
            }
        )

    action_counts = {"매수": 0, "관망": 0, "매도": 0}
    for asset in assets:
        signal = asset.get("signal", {})
        action = signal.get("cw_signal") or signal.get("signal") or "관망"
        action_counts[action] = action_counts.get(action, 0) + 1

    base_pct = round((1.0 - regime_tilt) * 100, 0)
    tilt_pct = round(regime_tilt * 100, 0)
    regime_method = "P^3 전이확률 기대배분" if tea_payload else "확률가중 국면 틸트"
    method = f"안정성 검증 포트폴리오 앙상블 {base_pct:.0f}% + {regime_method} {tilt_pct:.0f}% + 로보/Alpha Decay 미세조정 + 리밸런싱: 밴드(±5%p/상대25%)·국면전환 70% 부분이동"
    if regime_alpha.get("status") not in (None, "자료없음", "중립"):
        method += " + 국면별 Alpha Decay 검증 조정"

    return {
        "title": "국면 기반 완성형 추천 포트폴리오",
        "regime": dict(snapshot),
        "validity": validity,
        "automation": auto,
        "construction": {
            "method": method,
            "components": components or META_COMPONENTS,
            "base_weights": {key: round(value * 100, 1) for key, value in base_weights.items()},
            "regime_probability_blend": {
                key: round(value * 100, 1) for key, value in regime_blend.items()
            },
            "regime_target_source": (
                "P^3 전이행렬 기반 기대배분"
                if tea_payload else "현재 국면 확률 70% + 다음 분기 국면 확률 30%"
            ),
            "regime_target": {key: round(value * 100, 1) for key, value in regime_target.items()},
            "transition_expected_allocation": tea_payload,
            "pre_signal_weights": {
                key: round(value * 100, 1) for key, value in pre_signal_weights.items()
            },
            "signal_multipliers": signal_multipliers,
            "regime_alpha_signal_adjustment": regime_alpha,
        },
        "weights": {key: round(value * 100, 1) for key, value in final_weights.items()},
        "assets": assets,
        "action_counts": action_counts,
        "message": "국면은 자산 비중을 정하고, 로보신호와 Alpha Decay는 구성 자산의 진입/청산 시점을 보조합니다.",
        "caveat": "투자 조언이 아닌 분석용 결과입니다. ETF 분배금/세금/거래비용은 별도 고려가 필요합니다.",
    }


_portfolio_validity = portfolio_validity
_regime_probability_blend = regime_probability_blend
_probability_weighted_regime_target = probability_weighted_regime_target
_auto_regime_tilt = auto_regime_tilt
_signal_weight_multiplier = signal_weight_multiplier
_apply_signal_tilt = apply_signal_tilt
_default_asset_signal = default_asset_signal
_stock_analysis_to_asset_signal = stock_analysis_to_asset_signal
_load_regime_alpha_summary = load_regime_alpha_summary
_regime_alpha_signal_adjustment = regime_alpha_signal_adjustment
_build_recommendation_report = build_recommendation_report


