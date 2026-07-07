"""Robo-signal scoring helpers."""

from __future__ import annotations

import numpy as np

from kq_tool.analyzer.alpha_decay import classify_decay_quality, half_life_to_confidence
from kq_tool.config import (
    ROBO_BUY_THRESHOLD,
    ROBO_SELL_THRESHOLD,
    ROBO_SIGNAL_LABELS,
    ROBO_SIGNAL_WEIGHTS,
    SIGNAL_DIRECTION,
)

BASE_WEIGHT = dict(ROBO_SIGNAL_WEIGHTS)
SIGNAL_LABELS = list(ROBO_SIGNAL_LABELS)


def score_to_signal(score: float) -> str:
    """Map a 0-100 robo score to a user-facing signal."""

    if score >= ROBO_BUY_THRESHOLD:
        return "매수"
    if score <= ROBO_SELL_THRESHOLD:
        return "매도"
    return "관망"


def legacy_robo_score(s_rsi: int, s_macd: int, s_bb: int, s_ma20: int, s_ma60: int) -> float:
    """Compute the original threshold-based robo score."""

    sub_values = {
        "s_rsi": s_rsi,
        "s_macd": s_macd,
        "s_bb": s_bb,
        "s_ma20": s_ma20,
        "s_ma60": s_ma60,
    }
    raw = sum(float(sub_values[key]) * weight for key, weight in BASE_WEIGHT.items())
    return round(max(0, min(100, (raw + 80) / 1.6)), 1)


def confidence_weighted_robo(
    alpha: dict[str, dict],
    s_rsi: int,
    s_macd: int,
    s_bb: int,
    s_ma20: int,
    s_ma60: int,
) -> dict:
    """Compute Alpha-Decay confidence-weighted robo score and validity window."""

    sub_values = {
        "s_rsi": s_rsi,
        "s_macd": s_macd,
        "s_bb": s_bb,
        "s_ma20": s_ma20,
        "s_ma60": s_ma60,
    }

    active_by_sub: dict[str, list[tuple[str, dict]]] = {}
    for signal_name, (direction, subkey) in SIGNAL_DIRECTION.items():
        detail = alpha.get(signal_name)
        if not detail or not detail.get("is_active"):
            continue
        if sub_values.get(subkey) == direction:
            active_by_sub.setdefault(subkey, []).append((signal_name, detail))

    detail_rows = []
    weighted_sum = 0.0
    decay_days = []
    decay_weights = []

    for subkey, label in SIGNAL_LABELS:
        direction = sub_values[subkey]
        base_weight = BASE_WEIGHT[subkey]
        matches = active_by_sub.get(subkey, [])

        if direction == 0:
            detail_rows.append(
                {
                    "key": subkey,
                    "label": label,
                    "direction": 0,
                    "confidence": None,
                    "half_life": None,
                    "weight": 0.0,
                    "signal_name": None,
                    "decay_state": "neutral",
                    "decay_basis": "neutral",
                    "decay_reason_code": "neutral_indicator_signal",
                    "decay_reason_data": {},
                }
            )
            continue

        if matches:
            signal_name, alpha_detail = matches[0]
            decay_quality = classify_decay_quality(signal_name, alpha_detail)
            confidence = float(decay_quality["decay_confidence"])
            half_life = decay_quality["effective_half_life"]
            decay_basis = decay_quality["decay_basis"]
            if half_life and decay_quality["decay_state"] != "neutral":
                decay_days.append(half_life)
                decay_weight = confidence if decay_quality["decay_state"] == "measured" else confidence * 0.75
                decay_weights.append(decay_weight)
        else:
            decay_quality = {
                "decay_state": "neutral",
                "decay_basis": "neutral",
                "decay_reason_code": "no_active_alpha_signal",
                "decay_reason_data": {},
                "decay_family": None,
                "effective_half_life": None,
                "raw_half_life": None,
                "decay_confidence": 5.0,
                "min_decay_events": None,
            }
            confidence = 5.0
            signal_name = None
            half_life = None
            decay_basis = "neutral"

        effective_weight = base_weight * (confidence / 10.0)
        weighted_sum += direction * effective_weight

        detail_rows.append(
            {
                "key": subkey,
                "label": label,
                "direction": direction,
                "confidence": confidence,
                "half_life": half_life,
                "weight": round(effective_weight, 2),
                "signal_name": signal_name,
                "decay_basis": decay_basis,
                "decay_state": decay_quality["decay_state"],
                "decay_reason_code": decay_quality["decay_reason_code"],
                "decay_reason_data": decay_quality.get("decay_reason_data", {}),
                "decay_family": decay_quality.get("decay_family"),
                "raw_half_life": decay_quality.get("raw_half_life"),
                "min_decay_events": decay_quality.get("min_decay_events"),
            }
        )

    score = round(max(0, min(100, (weighted_sum + 80) / 1.6)), 1)
    signal = score_to_signal(score)

    active_confidences = [
        row["confidence"]
        for row in detail_rows
        if row["confidence"] is not None and row["direction"] != 0
    ]
    active_weights = [row["weight"] for row in detail_rows if row["direction"] != 0]
    overall_confidence = (
        round(float(np.average(active_confidences, weights=active_weights)), 1)
        if active_confidences and sum(active_weights) > 0
        else None
    )

    active_decay_states = [
        row.get("decay_state")
        for row in detail_rows
        if row["direction"] != 0 and row.get("decay_state") is not None
    ]
    if "measured" in active_decay_states:
        overall_decay_state = "measured"
    elif "imputed" in active_decay_states:
        overall_decay_state = "imputed"
    else:
        overall_decay_state = "neutral"

    exit_days = (
        round(float(np.average(decay_days, weights=decay_weights)), 1)
        if decay_days and sum(decay_weights) > 0
        else None
    )
    if exit_days is not None:
        valid_days = exit_days
        validity_basis = "alpha_decay"
        states = {row.get("decay_state") for row in detail_rows if row["direction"] != 0}
        if states == {"imputed"}:
            validity_text = f"계열 prior 기반 보수적 유효기간 ({exit_days:.0f}일)"
        elif "imputed" in states:
            validity_text = f"측정 반감기 + 계열 prior 종합 유효기간 ({exit_days:.0f}일)"
        else:
            validity_text = f"활성 신호의 Alpha Decay 기반 유효기간 (반감기 종합, {exit_days:.0f}일)"
    else:
        valid_days = 5 if signal in ("매수", "매도") else 10
        validity_basis = "review_interval"
        validity_text = "활성 신호 없음, 보수적 재점검 기간 적용 (5~10일)"

    return {
        "score": score,
        "signal": signal,
        "confidence": overall_confidence,
        "exit_days": exit_days,
        "decay_state": overall_decay_state,
        "valid_days": valid_days,
        "validity_basis": validity_basis,
        "validity_text": validity_text,
        "detail": detail_rows,
    }


_confidence_weighted_robo = confidence_weighted_robo
