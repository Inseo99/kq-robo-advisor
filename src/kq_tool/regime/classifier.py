"""Regime snapshot helpers."""

from __future__ import annotations

from typing import Any


def current_regime_snapshot(
    regime_model: Any,
    fallback_current: str = "리플레이션",
) -> dict:
    """Normalize current-regime model output for portfolio recommendation."""

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
