"""Band and regime-triggered rebalancing policy.

This layer answers a narrower question than portfolio construction: given the
recommended target weights, should we actually trade this month? The default
parameters are pre-registered policy values, not performance-tuned knobs:

- monthly check cadence
- rebalance band: min(absolute 5%p, relative 25% of target weight)
- band restoration: half-way between the violated band edge and target
- official regime transition: move 70% of the way from current weights to target

`reeval_flag` from the regime UI payload is deliberately recorded as a review
note only. It does not trigger a trade by itself, which avoids turning every
probability wobble into turnover.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


WeightMap = Mapping[str, float]


@dataclass(frozen=True)
class RebalancePolicy:
    """Pre-registered rebalancing rule set.

    Do not tune these values after looking at strategy performance. If a new
    policy is tested, count it as another trial in the validation layer.
    """

    abs_band: float = 0.05
    rel_band: float = 0.25
    restore_fraction: float = 0.50
    regime_shift_fraction: float = 0.70
    min_trade_weight: float = 0.0005


@dataclass(frozen=True)
class RebalanceDecision:
    """One monthly rebalancing decision."""

    action: str
    trigger: str
    target_weights: dict[str, float]
    turnover: float
    notes: tuple[str, ...]
    band_status: tuple[dict[str, object], ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "action": self.action,
            "trigger": self.trigger,
            "target_weights": dict(self.target_weights),
            "turnover": self.turnover,
            "notes": list(self.notes),
            "band_status": list(self.band_status),
        }


def normalize_weights(weights: WeightMap) -> dict[str, float]:
    """Return non-negative weights scaled to sum to 1 when possible."""

    cleaned = {str(k): max(0.0, float(v)) for k, v in dict(weights).items()}
    total = sum(cleaned.values())
    if total <= 0:
        return {k: 0.0 for k in cleaned}
    return {k: v / total for k, v in cleaned.items()}


def _all_assets(*maps: WeightMap) -> list[str]:
    seen: dict[str, None] = {}
    for mapping in maps:
        for key in mapping:
            seen[str(key)] = None
    return list(seen)


def _threshold(target_weight: float, policy: RebalancePolicy) -> float:
    target = max(0.0, float(target_weight))
    relative = target * policy.rel_band if target > 0 else policy.abs_band
    return min(policy.abs_band, relative) if target > 0 else policy.abs_band


def band_status(
    current_weights: WeightMap,
    target_weights: WeightMap,
    policy: RebalancePolicy | None = None,
) -> tuple[dict[str, object], ...]:
    """Return per-asset band status rows for UI/audit use."""

    policy = policy or RebalancePolicy()
    current = normalize_weights(current_weights)
    target = normalize_weights(target_weights)
    rows: list[dict[str, object]] = []

    for asset in _all_assets(current, target):
        cur = float(current.get(asset, 0.0))
        tgt = float(target.get(asset, 0.0))
        threshold = _threshold(tgt, policy)
        lower = max(0.0, tgt - threshold)
        upper = tgt + threshold
        drift = cur - tgt
        if cur > upper:
            state = "over"
        elif cur < lower:
            state = "under"
        else:
            state = "inside"
        rows.append(
            {
                "asset": asset,
                "current_weight": cur,
                "target_weight": tgt,
                "drift": drift,
                "threshold": threshold,
                "lower": lower,
                "upper": upper,
                "state": state,
                "breached": state != "inside",
            }
        )
    return tuple(rows)


def turnover(from_weights: WeightMap, to_weights: WeightMap) -> float:
    """One-way portfolio turnover needed to move from one weight map to another."""

    from_w = normalize_weights(from_weights)
    to_w = normalize_weights(to_weights)
    assets = _all_assets(from_w, to_w)
    return 0.5 * sum(abs(float(to_w.get(a, 0.0)) - float(from_w.get(a, 0.0))) for a in assets)


def _partial_regime_move(current: WeightMap, target: WeightMap, fraction: float) -> dict[str, float]:
    cur = normalize_weights(current)
    tgt = normalize_weights(target)
    out = {}
    for asset in _all_assets(cur, tgt):
        out[asset] = float(cur.get(asset, 0.0)) + fraction * (
            float(tgt.get(asset, 0.0)) - float(cur.get(asset, 0.0))
        )
    return normalize_weights(out)


def _band_restore(current: WeightMap, target: WeightMap, rows: tuple[dict[str, object], ...], policy: RebalancePolicy) -> dict[str, float]:
    cur = normalize_weights(current)
    tgt = normalize_weights(target)
    desired = dict(cur)
    locked: set[str] = set()

    for row in rows:
        if not row["breached"]:
            continue
        asset = str(row["asset"])
        target_weight = float(row["target_weight"])
        if row["state"] == "over":
            boundary = float(row["upper"])
        elif row["state"] == "under":
            boundary = float(row["lower"])
        else:
            continue
        desired[asset] = boundary + policy.restore_fraction * (target_weight - boundary)
        locked.add(asset)

    # Preserve the exact restored weights and allocate the residual across the
    # untouched assets in proportion to their strategic targets.
    residual = max(0.0, 1.0 - sum(desired.get(asset, 0.0) for asset in locked))
    free_assets = [asset for asset in _all_assets(cur, tgt) if asset not in locked]
    free_target_sum = sum(float(tgt.get(asset, 0.0)) for asset in free_assets)
    if free_assets:
        if free_target_sum > 0:
            for asset in free_assets:
                desired[asset] = residual * float(tgt.get(asset, 0.0)) / free_target_sum
        else:
            equal = residual / len(free_assets)
            for asset in free_assets:
                desired[asset] = equal
    return normalize_weights(desired)


def decide(
    current_weights: WeightMap,
    target_weights: WeightMap,
    *,
    previous_regime: str | None = None,
    current_regime: str | None = None,
    reeval_flag: bool = False,
    policy: RebalancePolicy | None = None,
) -> RebalanceDecision:
    """Return the monthly trade/no-trade decision.

    Trigger priority is fixed as:
    T2 official regime transition > T1 band breach > HOLD.
    """

    policy = policy or RebalancePolicy()
    current = normalize_weights(current_weights)
    target = normalize_weights(target_weights)
    rows = band_status(current, target, policy)
    notes: list[str] = []

    if reeval_flag:
        notes.append("조기 재평가 플래그 감지: 점검 사유로만 기록, 단독 거래 없음")

    regime_changed = (
        previous_regime is not None
        and current_regime is not None
        and str(previous_regime) != str(current_regime)
    )
    if regime_changed:
        new_weights = _partial_regime_move(current, target, policy.regime_shift_fraction)
        return RebalanceDecision(
            action="trade",
            trigger="T2_REGIME_CHANGE",
            target_weights=new_weights,
            turnover=turnover(current, new_weights),
            notes=tuple([f"공식 국면 전환: {previous_regime} -> {current_regime}", *notes]),
            band_status=rows,
        )

    breached = [row for row in rows if bool(row["breached"])]
    if breached:
        new_weights = _band_restore(current, target, rows, policy)
        return RebalanceDecision(
            action="trade",
            trigger="T1_BAND_BREACH",
            target_weights=new_weights,
            turnover=turnover(current, new_weights),
            notes=tuple([f"밴드 이탈 {len(breached)}개 자산 부분 복원", *notes]),
            band_status=rows,
        )

    return RebalanceDecision(
        action="hold",
        trigger="NO_TRADE",
        target_weights=current,
        turnover=0.0,
        notes=tuple(notes or ["밴드 내부: 거래 없음"]),
        band_status=rows,
    )


__all__ = [
    "RebalanceDecision",
    "RebalancePolicy",
    "band_status",
    "decide",
    "normalize_weights",
    "turnover",
]
