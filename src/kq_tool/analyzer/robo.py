"""Robo-signal scoring helpers."""

from __future__ import annotations

import numpy as np

from kq_tool.analyzer.alpha_decay import half_life_to_confidence
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
                }
            )
            continue

        if matches:
            signal_name, alpha_detail = matches[0]
            confidence = half_life_to_confidence(
                alpha_detail.get("half_life"), alpha_detail.get("count", 0)
            )
            half_life = alpha_detail.get("half_life")
            decay_basis = "half_life" if half_life else None
            if half_life:
                decay_days.append(half_life)
                decay_weights.append(confidence)
            else:
                horizon_returns = alpha_detail.get("horizon_rets", {})
                valid_returns = [
                    (int(horizon), float(ret))
                    for horizon, ret in horizon_returns.items()
                    if ret is not None and ret > 0
                ]
                if valid_returns:
                    peak_day, _ = max(valid_returns, key=lambda item: item[1])
                    decay_days.append(float(peak_day))
                    decay_weights.append(confidence * 0.7)
                    half_life = peak_day
                    decay_basis = "peak"
        else:
            confidence = 5.0
            signal_name = None
            half_life = None
            decay_basis = None

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

    exit_days = (
        round(float(np.average(decay_days, weights=decay_weights)), 1)
        if decay_days and sum(decay_weights) > 0
        else None
    )
    if exit_days is not None:
        valid_days = exit_days
        validity_basis = "alpha_decay"
        validity_text = f"활성 신호의 Alpha Decay 기반 유효기간 (반감기/peak 종합, {exit_days:.0f}일)"
    else:
        valid_days = 5 if signal in ("매수", "매도") else 10
        validity_basis = "review_interval"
        validity_text = "활성 신호 없음, 보수적 재점검 기간 적용 (5~10일)"

    return {
        "score": score,
        "signal": signal,
        "confidence": overall_confidence,
        "exit_days": exit_days,
        "valid_days": valid_days,
        "validity_basis": validity_basis,
        "validity_text": validity_text,
        "detail": detail_rows,
    }


_confidence_weighted_robo = confidence_weighted_robo
