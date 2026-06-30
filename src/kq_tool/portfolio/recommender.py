"""Recommendation portfolio helper rules."""

from __future__ import annotations

from collections.abc import Mapping

from kq_tool.portfolio.validity import portfolio_validity
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


def signal_weight_multiplier(signal: Mapping[str, object]) -> float:
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
        multiplier = 1.0 + 0.10 * strength
    elif action == "매도":
        multiplier = 1.0 - 0.12 * strength
    else:
        multiplier = 1.0

    return round(float(max(0.75, min(1.12, multiplier))), 4)


def apply_signal_tilt(
    weights: Mapping[str, float],
    signal_map: Mapping[str, Mapping[str, object]],
) -> tuple[dict[str, float], dict[str, float]]:
    """Apply robo signal multipliers and renormalize portfolio weights."""

    adjusted = {}
    multipliers = {}
    for ticker, weight in weights.items():
        multiplier = signal_weight_multiplier(signal_map.get(ticker, {}))
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
    regime_target: Mapping[str, float],
    signal_map: Mapping[str, Mapping[str, object]],
    etf_meta: Mapping[str, tuple[str, str, str]],
    components: list[Mapping[str, object]] | None = None,
) -> dict:
    """Build the complete recommendation portfolio payload."""

    auto = auto_regime_tilt(snapshot)
    regime_tilt = float(auto["regime_tilt"])
    pre_signal_weights = combine_weight_sets(
        [(1.0 - regime_tilt, base_weights), (regime_tilt, regime_target)]
    )
    validity = portfolio_validity(snapshot)
    final_weights, signal_multipliers = apply_signal_tilt(pre_signal_weights, signal_map)

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
    return {
        "title": "국면 기반 완성형 추천 포트폴리오",
        "regime": dict(snapshot),
        "validity": validity,
        "automation": auto,
        "construction": {
            "method": f"안정성 검증 포트폴리오 앙상블 {base_pct:.0f}% + 현재 국면 틸트 {tilt_pct:.0f}% + 로보/Alpha Decay 미세조정",
            "components": components or META_COMPONENTS,
            "base_weights": {key: round(value * 100, 1) for key, value in base_weights.items()},
            "regime_target": {key: round(value * 100, 1) for key, value in regime_target.items()},
            "pre_signal_weights": {
                key: round(value * 100, 1) for key, value in pre_signal_weights.items()
            },
            "signal_multipliers": signal_multipliers,
        },
        "weights": {key: round(value * 100, 1) for key, value in final_weights.items()},
        "assets": assets,
        "action_counts": action_counts,
        "message": "국면은 자산 비중을 정하고, 로보신호와 Alpha Decay는 구성 자산의 진입/청산 시점을 보조합니다.",
        "caveat": "투자 조언이 아닌 분석용 결과입니다. ETF 분배금/세금/거래비용은 별도 고려가 필요합니다.",
    }


_portfolio_validity = portfolio_validity
_auto_regime_tilt = auto_regime_tilt
_signal_weight_multiplier = signal_weight_multiplier
_apply_signal_tilt = apply_signal_tilt
_default_asset_signal = default_asset_signal
_stock_analysis_to_asset_signal = stock_analysis_to_asset_signal
_build_recommendation_report = build_recommendation_report
