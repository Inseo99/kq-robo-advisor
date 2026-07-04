"""Regime API response helpers."""

from __future__ import annotations

from typing import Any


def build_regime_ai_payload(
    regime_model: Any,
    fallback_payload: dict,
    *,
    regime_desc: dict | None = None,
    regime_order: list[str] | tuple[str, ...] | None = None,
) -> dict:
    """Build the `/api/regime_ai` payload from a trained model or fallback."""

    if regime_model is None or not getattr(regime_model, "trained", False):
        fallback = dict(fallback_payload)
        fallback["model_type"] = "Rule-based (모델 미학습)"
        fallback["model_status"] = "not_trained"
        return fallback

    result = regime_model.predict_current()
    if result is None:
        raise RuntimeError("예측 실패")

    payload = dict(result)
    if regime_desc is not None:
        payload["regime_desc"] = regime_desc
    if regime_order is not None:
        payload["regime_order"] = list(regime_order)
    payload["model_status"] = "trained"
    payload["sample_count"] = (
        len(regime_model.train_data)
        if getattr(regime_model, "train_data", None) is not None
        else 0
    )
    return payload


_build_regime_ai_payload = build_regime_ai_payload
