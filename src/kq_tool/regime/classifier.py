"""Regime snapshot helpers."""

from __future__ import annotations

from typing import Any

from .ui_payload import build_payload


def _snapshot_from_ui_payload() -> dict:
    ui = build_payload()
    current = ui["current_regime"]
    diagnostics = ui.get("diagnostics") or {}
    return {
        "current": current,
        "current_regime": current,
        "confidence": float((ui.get("nowcast_probs") or {}).get(current, 0.0)),
        "probs": ui["nowcast_probs"],
        "next_quarter": ui["next_quarter_probs"],
        "duration_avg": diagnostics.get("avg_duration_by_regime_q", {}),
        "model_type": ui["method_label"],
        "method_label": ui["method_label"],
        "expected_remaining_days": ui["expected_remaining_days"],
        "validity_display": ui["validity_display"],
        "reeval_flag": ui["reeval_flag"],
        "reeval_reasons": ui["reeval_reasons"],
        "stay_prob_next_quarter": ui["stay_prob_next_quarter"],
    }


def current_regime_snapshot(
    regime_model: Any,
    fallback_current: str = "리플레이션",
) -> dict:
    """Normalize current-regime output for portfolio recommendation.

    Prefer the validated v2 CSV payload so recommendation validity, regime
    probabilities, and UI displays all share one source of truth.
    """

    try:
        return _snapshot_from_ui_payload()
    except Exception:
        pass

    try:
        if regime_model is not None and getattr(regime_model, "trained", False):
            result = regime_model.predict_current()
            probs = result.get("probs", {}) or {}
            current = result.get("current_regime") or (
                max(probs, key=probs.get) if probs else fallback_current
            )
            confidence = float(probs.get(current, 0.0)) if probs else 0.0
            next_quarter = result.get("next_quarter", {}) or {}
            duration_avg = result.get("duration_avg", {}) or {}
            model_type = result.get("model_type", "Rule-based")
        else:
            current = fallback_current
            probs = {current: 1.0}
            confidence = 1.0
            next_quarter = {}
            duration_avg = {}
            model_type = "Rule-based"
    except Exception:
        current = fallback_current
        probs = {current: 1.0}
        confidence = 1.0
        next_quarter = {}
        duration_avg = {}
        model_type = "Rule-based"

    return {
        "current": current,
        "confidence": confidence,
        "probs": probs,
        "next_quarter": next_quarter,
        "duration_avg": duration_avg,
        "model_type": model_type,
    }


_current_regime_snapshot = current_regime_snapshot
