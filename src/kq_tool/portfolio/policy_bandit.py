"""Shadow-ledger policy bandit.

This is deliberately SHADOW-ONLY.  It compares rebalancing policies that are
already tracked in the shadow ledger and produces an audit payload.  It never
places trades or changes the adopted policy by itself.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

GLOBAL_CONTEXT = "__global__"


@dataclass(frozen=True)
class BanditConfig:
    eta: float = 2.0
    gamma: float = 0.97
    n0: float = 12.0
    eps_floor: float = 0.05
    shadow_only: bool = True


@dataclass
class BanditResult:
    probs: pd.DataFrame
    chosen: pd.Series
    meta_returns: pd.Series
    static_cum: pd.Series
    bandit_cum: float
    best_static: str
    regret: float
    regime_weights_final: pd.DataFrame
    n_by_regime: pd.Series
    config: dict[str, Any] = field(default_factory=dict)
    fp_json: str = ""

    def to_payload(self) -> dict[str, Any]:
        return {
            "shadow_only": bool(self.config.get("shadow_only", True)),
            "current_recommendation": {
                "by_regime": {
                    str(regime): self.regime_weights_final.loc[regime].round(4).to_dict()
                    for regime in self.regime_weights_final.index
                },
                "n_by_regime": {str(k): int(v) for k, v in self.n_by_regime.items()},
            },
            "performance": {
                "bandit_cum": round(float(self.bandit_cum), 6),
                "static_cum": {str(k): round(float(v), 6) for k, v in self.static_cum.items()},
                "best_static": self.best_static,
                "regret_vs_best": round(float(self.regret), 6),
            },
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
        if isinstance(value, dict):
            return {str(k): norm(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
        return value

    return hashlib.sha256(json.dumps(norm(obj), ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def normalize_rewards(values: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    mu = float(values.mean())
    sd = float(values.std(ddof=0))
    return (values - mu) / (sd + eps)


def softmax(scores: np.ndarray, eta: float) -> np.ndarray:
    z = eta * (scores - scores.max())
    weights = np.exp(z)
    return weights / weights.sum()


def _align_inputs(rewards: pd.DataFrame, regimes: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    r = rewards.copy()
    g = regimes.copy()
    r.index = pd.to_datetime(r.index).to_period("M").to_timestamp("M")
    g.index = pd.to_datetime(g.index).to_period("M").to_timestamp("M")
    common = r.index.intersection(g.index)
    if len(common) < 2:
        raise ValueError("섀도 원장과 국면 이력의 기간 정합 실패")
    r = r.loc[common].sort_index().apply(pd.to_numeric, errors="coerce")
    g = g.loc[common].sort_index().astype(str)
    if r.isna().any().any():
        raise ValueError("섀도 원장 수익률에 결측이 있습니다")
    return r, g


def run_shadow_bandit(
    rewards: pd.DataFrame,
    regimes: pd.Series,
    config: BanditConfig | None = None,
) -> BanditResult:
    cfg = config or BanditConfig()
    rewards, regimes = _align_inputs(rewards, regimes)

    policies = list(rewards.columns)
    n_policies = len(policies)
    contexts = [GLOBAL_CONTEXT] + sorted(regimes.unique().tolist())
    scores = {context: np.zeros(n_policies) for context in contexts}
    counts = {context: 0.0 for context in contexts}

    probs_rows: dict[pd.Timestamp, pd.Series] = {}
    chosen_rows: dict[pd.Timestamp, str] = {}
    meta_rows: dict[pd.Timestamp, float] = {}

    for date, row in rewards.iterrows():
        context = str(regimes.loc[date])
        global_probs = softmax(scores[GLOBAL_CONTEXT], cfg.eta)
        context_probs = softmax(scores[context], cfg.eta)
        kappa = counts[context] / (counts[context] + cfg.n0)
        probs = kappa * context_probs + (1.0 - kappa) * global_probs
        probs = (1.0 - cfg.eps_floor) * probs + cfg.eps_floor / n_policies

        probs_rows[date] = pd.Series(probs, index=policies)
        chosen_rows[date] = policies[int(np.argmax(probs))]
        meta_rows[date] = float(np.dot(probs, row.values))

        reward_norm = normalize_rewards(row.values.astype(float))
        for ctx in (GLOBAL_CONTEXT, context):
            scores[ctx] = cfg.gamma * scores[ctx] + reward_norm
        counts[context] += 1.0
        counts[GLOBAL_CONTEXT] += 1.0

    probs_df = pd.DataFrame(probs_rows).T[policies]
    meta = pd.Series(meta_rows).sort_index()
    static_cum = rewards.sum(axis=0)
    bandit_cum = float(meta.sum())
    best_static = str(static_cum.idxmax())

    final_global = softmax(scores[GLOBAL_CONTEXT], cfg.eta)
    final_rows = {}
    for context in contexts[1:]:
        kappa = counts[context] / (counts[context] + cfg.n0)
        context_probs = softmax(scores[context], cfg.eta)
        probs = kappa * context_probs + (1.0 - kappa) * final_global
        probs = (1.0 - cfg.eps_floor) * probs + cfg.eps_floor / n_policies
        final_rows[context] = pd.Series(probs, index=policies)

    return BanditResult(
        probs=probs_df,
        chosen=pd.Series(chosen_rows).sort_index(),
        meta_returns=meta,
        static_cum=static_cum,
        bandit_cum=bandit_cum,
        best_static=best_static,
        regret=float(static_cum.max() - bandit_cum),
        regime_weights_final=pd.DataFrame(final_rows).T[policies],
        n_by_regime=pd.Series({ctx: counts[ctx] for ctx in contexts[1:]}, dtype=float).astype(int),
        config=asdict(cfg),
        fp_json=_fingerprint({"rewards": rewards, "regimes": regimes, "config": asdict(cfg)}),
    )


def context_free_cum(rewards: pd.DataFrame, config: BanditConfig | None = None) -> float:
    cfg = config or BanditConfig()
    rewards = rewards.copy().apply(pd.to_numeric, errors="coerce").dropna()
    n_policies = rewards.shape[1]
    scores = np.zeros(n_policies)
    cumulative = 0.0
    for _, row in rewards.iterrows():
        probs = softmax(scores, cfg.eta)
        probs = (1.0 - cfg.eps_floor) * probs + cfg.eps_floor / n_policies
        cumulative += float(np.dot(probs, row.values))
        scores = cfg.gamma * scores + normalize_rewards(row.values.astype(float))
    return cumulative

