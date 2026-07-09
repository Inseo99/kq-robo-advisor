"""Transition-Expected Allocation.

현재 국면 하나만으로 배분하지 않고, 검증된 월간 전이행렬 P를 P^h로 전개해
다음 국면 분포의 기대 배분을 계산한다.

    pi_h = pi_0 @ P^h
    w_expected = pi_h @ W_regime

유지확률이 충분히 높으면 현재 국면 배분을 그대로 유지해 불필요한 회전을
줄인다.  추천 탭에서는 h=3을 기본값으로 써서 기존 "다음 분기 P^3" 설명과
정합시킨다.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TEAConfig:
    horizon: int = 3
    p_stay_lo: float = 0.50
    p_stay_hi: float = 0.80
    always_blend: bool = False
    renormalize: bool = True


@dataclass
class TEAResult:
    weights: pd.Series
    weights_current: pd.Series
    weights_expected: pd.Series
    pi0: pd.Series
    pi_h: pd.Series
    p_stay: float
    alpha: float
    config: dict[str, Any] = field(default_factory=dict)
    fp_json: str = ""

    def to_payload(self) -> dict[str, Any]:
        return {
            "weights": self.weights.round(6).to_dict(),
            "weights_current": self.weights_current.round(6).to_dict(),
            "weights_expected": self.weights_expected.round(6).to_dict(),
            "pi0": self.pi0.round(6).to_dict(),
            "pi_h": self.pi_h.round(6).to_dict(),
            "p_stay": round(float(self.p_stay), 6),
            "alpha": round(float(self.alpha), 6),
            "config": self.config,
            "fp_json": self.fp_json,
        }


def _fingerprint(obj: Any) -> str:
    def norm(value: Any) -> Any:
        if isinstance(value, pd.DataFrame):
            return {
                "type": "df",
                "shape": list(value.shape),
                "index": [str(value.index[0]), str(value.index[-1])] if len(value) else [],
                "columns": list(map(str, value.columns)),
                "sha": hashlib.sha256(
                    pd.util.hash_pandas_object(value, index=True).values.tobytes()
                ).hexdigest(),
            }
        if isinstance(value, pd.Series):
            return norm(value.to_frame())
        if isinstance(value, np.ndarray):
            arr = np.ascontiguousarray(value)
            return {
                "type": "ndarray",
                "shape": list(arr.shape),
                "sha": hashlib.sha256(arr.tobytes()).hexdigest(),
            }
        if isinstance(value, Mapping):
            return {str(k): norm(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
        if isinstance(value, (list, tuple)):
            return [norm(v) for v in value]
        return value

    payload = json.dumps(norm(obj), ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _validate_transition_matrix(matrix: pd.DataFrame) -> None:
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"전이행렬이 정방행렬이 아닙니다: {matrix.shape}")
    if not matrix.index.equals(matrix.columns):
        raise ValueError("전이행렬 index/columns 라벨이 다릅니다")
    if (matrix.values < -1e-10).any():
        raise ValueError("전이행렬에 음수 확률이 있습니다")
    row_sums = matrix.sum(axis=1)
    if not np.allclose(row_sums.values, 1.0, atol=1e-6):
        raise ValueError(f"전이행렬 행합이 1이 아닙니다: {row_sums.to_dict()}")


def _validate_regime_weights(weights: pd.DataFrame) -> None:
    row_sums = weights.sum(axis=1)
    if not np.allclose(row_sums.values, 1.0, atol=1e-6):
        raise ValueError(f"국면별 배분 행합이 1이 아닙니다: {row_sums.to_dict()}")
    if (weights.values < -1e-10).any():
        raise ValueError("국면별 기준 배분에 음수 비중이 있습니다")


def normalize_probability_series(probs: pd.Series) -> pd.Series:
    clean = probs.astype(float).clip(lower=0.0).fillna(0.0)
    total = float(clean.sum())
    if total <= 0:
        raise ValueError("확률 합이 0입니다")
    return clean / total


def transition_matrix_from_payload(payload_matrix: Mapping[str, Any]) -> pd.DataFrame:
    """Convert ui_payload transition_matrix into a numeric DataFrame."""

    regimes = list(payload_matrix.keys())
    values: dict[str, dict[str, float]] = {}
    for before in regimes:
        values[before] = {}
        row = payload_matrix.get(before, {}) or {}
        for after in regimes:
            cell = row.get(after, 0.0) if isinstance(row, Mapping) else 0.0
            if isinstance(cell, Mapping):
                cell = cell.get("prob", 0.0)
            values[before][after] = float(cell or 0.0)
    matrix = pd.DataFrame(values).T.reindex(index=regimes, columns=regimes).astype(float)
    matrix = matrix.div(matrix.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    _validate_transition_matrix(matrix)
    return matrix


def regime_weights_frame(regime_targets: Mapping[str, Mapping[str, float]]) -> pd.DataFrame:
    frame = pd.DataFrame(regime_targets).T.fillna(0.0).astype(float)
    frame = frame.div(frame.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    _validate_regime_weights(frame)
    return frame


def compute_transition_expected_allocation(
    transition_matrix: pd.DataFrame,
    regime_weights: pd.DataFrame,
    current_probs: pd.Series,
    config: TEAConfig | None = None,
) -> TEAResult:
    cfg = config or TEAConfig()
    _validate_transition_matrix(transition_matrix)
    _validate_regime_weights(regime_weights)

    regimes = list(transition_matrix.index)
    weights = regime_weights.reindex(regimes)
    if weights.isna().any().any():
        raise ValueError("regime_weights에 전이행렬 국면과 매칭되지 않는 행이 있습니다")

    pi0 = normalize_probability_series(current_probs.reindex(regimes).fillna(0.0))
    if cfg.horizon < 1:
        raise ValueError("horizon은 1 이상이어야 합니다")

    ph = np.linalg.matrix_power(transition_matrix.values.astype(float), cfg.horizon)
    pi_h = pd.Series(pi0.values @ ph, index=regimes, name="pi_h")
    pi_h = normalize_probability_series(pi_h)
    p_stay = float(np.dot(pi0.values, np.diag(ph)))

    if cfg.always_blend:
        alpha = 1.0
    else:
        span = max(float(cfg.p_stay_hi - cfg.p_stay_lo), 1e-12)
        alpha = float(np.clip((cfg.p_stay_hi - p_stay) / span, 0.0, 1.0))

    weights_current = pd.Series(pi0.values @ weights.values, index=weights.columns)
    weights_expected = pd.Series(pi_h.values @ weights.values, index=weights.columns)
    final = (1.0 - alpha) * weights_current + alpha * weights_expected
    final = final.clip(lower=0.0)
    if cfg.renormalize:
        total = float(final.sum())
        if total <= 0:
            raise ValueError("최종 배분 합이 0입니다")
        final = final / total

    return TEAResult(
        weights=final,
        weights_current=weights_current,
        weights_expected=weights_expected,
        pi0=pi0,
        pi_h=pi_h,
        p_stay=p_stay,
        alpha=alpha,
        config=asdict(cfg),
        fp_json=_fingerprint(
            {
                "transition_matrix": transition_matrix,
                "regime_weights": weights,
                "current_probs": pi0,
                "config": asdict(cfg),
            }
        ),
    )


def transition_expected_regime_target(
    snapshot: Mapping[str, Any],
    regime_targets: Mapping[str, Mapping[str, float]],
    config: TEAConfig | None = None,
) -> TEAResult:
    """Build a transition-expected asset target from a regime snapshot."""

    if "transition_matrix" in snapshot:
        matrix = transition_matrix_from_payload(snapshot["transition_matrix"])
    else:
        regimes = list(regime_targets.keys())
        next_probs = snapshot.get("next_quarter") or {}
        current = str(snapshot.get("current") or snapshot.get("current_regime") or regimes[0])
        row = pd.Series({regime: float(next_probs.get(regime, 0.0)) for regime in regimes})
        if row.sum() <= 0 and current in regimes:
            row[current] = 1.0
        row = normalize_probability_series(row)
        matrix = pd.DataFrame(np.eye(len(regimes)), index=regimes, columns=regimes)
        matrix.loc[current] = row.reindex(regimes).values

    weights = regime_weights_frame(regime_targets)
    regimes = list(matrix.index)
    current_probs_raw = snapshot.get("probs") or snapshot.get("nowcast_probs") or {}
    current_probs = pd.Series({regime: float(current_probs_raw.get(regime, 0.0)) for regime in regimes})
    if current_probs.sum() <= 0:
        current = str(snapshot.get("current") or snapshot.get("current_regime") or "")
        current_probs = pd.Series(0.0, index=regimes)
        if current in current_probs.index:
            current_probs.loc[current] = 1.0
    return compute_transition_expected_allocation(matrix, weights, current_probs, config)

