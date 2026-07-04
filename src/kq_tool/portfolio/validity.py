"""Portfolio recommendation validity and review-window helpers."""

from __future__ import annotations

from collections.abc import Mapping


def portfolio_validity(snapshot: Mapping[str, object]) -> dict:
    """Estimate validity/review window for the current portfolio recommendation."""

    current = str(snapshot["current"])
    confidence = float(snapshot.get("confidence", 0.0) or 0.0)

    if snapshot.get("expected_remaining_days") is not None:
        valid_days = int(round(float(snapshot.get("expected_remaining_days") or 0)))
        stay_prob = snapshot.get("stay_prob_next_quarter")
        try:
            stay_prob = float(stay_prob) if stay_prob is not None else None
        except Exception:
            stay_prob = None
        reasons = snapshot.get("reeval_reasons") or []
        if isinstance(reasons, str):
            reasons = [reasons]
        review_rule = " / ".join(str(x) for x in reasons) if reasons else "월간 점검 / 전이확률 트리거 시 조기 재평가"
        return {
            "valid_days": valid_days,
            "valid_weeks": round(valid_days / 5, 1),
            "regime_duration_quarters": None,
            "confidence": round(confidence * 100, 1),
            "next_same_regime_prob": round(stay_prob * 100, 1) if stay_prob is not None else None,
            "review_rule": review_rule,
            "reeval_flag": bool(snapshot.get("reeval_flag", False)),
            "rebalance": "월간 점검 / 전이확률 트리거 시 조기 재평가",
        }

    duration_q = (snapshot.get("duration_avg") or {}).get(current)  # type: ignore[union-attr]
    try:
        duration_q = float(duration_q)
    except Exception:
        duration_q = None

    base_days = duration_q * 63 if duration_q and duration_q > 0 else 30
    confidence_factor = 0.55 + min(1.0, max(0.0, confidence)) * 0.60
    valid_days = int(round(max(10, min(90, base_days * confidence_factor))))

    stay_prob = None
    next_quarter = snapshot.get("next_quarter") or {}
    if current in next_quarter:  # type: ignore[operator]
        try:
            stay_prob = float(next_quarter[current])  # type: ignore[index]
        except Exception:
            stay_prob = None

    return {
        "valid_days": valid_days,
        "valid_weeks": round(valid_days / 5, 1),
        "regime_duration_quarters": round(duration_q, 2) if duration_q else None,
        "confidence": round(confidence * 100, 1),
        "next_same_regime_prob": round(stay_prob * 100, 1) if stay_prob is not None else None,
        "review_rule": "국면 확률 60% 이하 또는 다음 분기 전환확률 40% 이상이면 조기 재평가",
        "rebalance": "월간 점검 / 분기 리밸런싱 기본",
    }


_portfolio_validity = portfolio_validity

