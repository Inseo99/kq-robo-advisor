"""Regime API response helpers."""

from __future__ import annotations

from typing import Any

from .ui_payload import DEFAULT_REGIME_DESC, build_payload


def _build_ui_regime_payload(
    fallback_payload: dict,
    *,
    regime_desc: dict | None = None,
    regime_order: list[str] | tuple[str, ...] | None = None,
) -> dict:
    ui = build_payload()
    order = list(regime_order or ui["nowcast_probs"].keys())
    desc = regime_desc or DEFAULT_REGIME_DESC
    diagnostics = ui.get("diagnostics") or {}
    current_features = dict((fallback_payload or {}).get("current_features") or {})
    current_features.setdefault("gdp_growth", 0.0)
    current_features.setdefault("spread", 0.0)
    current_features.setdefault("usd_change", 0.0)

    return {
        "current": ui["current_regime"],
        "current_regime": ui["current_regime"],
        "probs": ui["nowcast_probs"],
        "next_quarter": ui["next_quarter_probs"],
        "transition_matrix": ui["transition_matrix"],
        "duration_avg": diagnostics.get("avg_duration_by_regime_q", {}),
        "regime_desc": desc,
        "regime_order": order,
        "model_type": ui["method_label"],
        "method_label": ui["method_label"],
        "model_status": "trained",
        "sample_count": diagnostics.get("n_months", 0),
        "current_features": current_features,
        "validity": {
            "valid_days": ui["expected_remaining_days"],
            "validity_display": ui["validity_display"],
            "next_same_regime_prob": round(ui["stay_prob_next_quarter"] * 100, 1),
            "reeval_flag": ui["reeval_flag"],
            "reeval_reasons": ui["reeval_reasons"],
        },
        "ui_payload": ui,
    }


def build_regime_ai_payload(
    regime_model: Any,
    fallback_payload: dict,
    *,
    regime_desc: dict | None = None,
    regime_order: list[str] | tuple[str, ...] | None = None,
) -> dict:
    """Build the `/api/regime_ai` payload from the validated regime verdicts.

    The validated v2 pipeline is preferred over the legacy in-memory model. If
    the CSV artifacts are missing, the old model/fallback path keeps the app
    runnable for team handoff.
    """

    try:
        return _build_ui_regime_payload(
            fallback_payload,
            regime_desc=regime_desc,
            regime_order=regime_order,
        )
    except Exception:
        pass

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
