r"""Compare band-triggered rebalancing cost against quarterly calendar rebalancing.

Run synthetic smoke test:
  python scripts\check_rebalancing_cost.py --cost-bps 15

Run on real monthly returns CSV:
  python scripts\check_rebalancing_cost.py --returns data\assets\monthly_returns.csv --cost-bps 15
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kq_tool.portfolio.rebalancing import RebalancePolicy, decide, normalize_weights, turnover

DEFAULT_TARGET = {"stocks": 0.40, "bonds": 0.35, "gold": 0.15, "cash": 0.10}


def synthetic_returns(n_months: int = 96, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2016-01-31", periods=n_months, freq="ME")
    means = np.array([0.0065, 0.0022, 0.0035, 0.0018])
    vols = np.array([0.055, 0.018, 0.040, 0.002])
    corr = np.array([
        [1.00, -0.15, 0.10, 0.00],
        [-0.15, 1.00, 0.05, 0.00],
        [0.10, 0.05, 1.00, 0.00],
        [0.00, 0.00, 0.00, 1.00],
    ])
    cov = corr * np.outer(vols, vols)
    arr = rng.multivariate_normal(means, cov, size=n_months)
    return pd.DataFrame(arr, index=idx, columns=list(DEFAULT_TARGET))


def load_returns(path: Path | None) -> pd.DataFrame:
    if path is None:
        return synthetic_returns()
    frame = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    missing = [col for col in DEFAULT_TARGET if col not in frame.columns]
    if missing:
        raise ValueError(f"missing required return columns: {missing}")
    return frame[list(DEFAULT_TARGET)].dropna()


def _drift(weights: dict[str, float], returns: pd.Series) -> tuple[dict[str, float], float]:
    gross = {asset: weights.get(asset, 0.0) * (1.0 + float(returns[asset])) for asset in weights}
    portfolio_return = sum(gross.values()) - 1.0
    return normalize_weights(gross), portfolio_return


def run_strategy(
    returns: pd.DataFrame,
    mode: str,
    *,
    target: dict[str, float] | None = None,
    cost_bps: float = 15.0,
    policy: RebalancePolicy | None = None,
) -> dict[str, float]:
    target = normalize_weights(target or DEFAULT_TARGET)
    policy = policy or RebalancePolicy()
    cost_rate = max(0.0, float(cost_bps)) / 10_000.0
    weights = dict(target)
    value = 1.0
    total_turnover = 0.0
    total_cost = 0.0

    for i, (_, row) in enumerate(returns.iterrows(), start=1):
        weights, ret = _drift(weights, row)
        value *= (1.0 + ret)

        new_weights = weights
        trade_turnover = 0.0
        mode_key = str(mode).strip().lower()
        if mode_key == "quarterly":
            if i % 3 != 0:
                pass
            else:
                new_weights = dict(target)
                trade_turnover = turnover(weights, new_weights)
        elif mode_key == "band":
            decision = decide(weights, target, policy=policy)
            if decision.action == "trade":
                new_weights = decision.target_weights
                trade_turnover = decision.turnover
        elif mode_key == "none":
            pass
        else:
            raise ValueError(f"unknown mode: {mode!r}")

        if trade_turnover > 0:
            cost = trade_turnover * cost_rate
            value *= (1.0 - cost)
            total_cost += cost
            total_turnover += trade_turnover
            weights = normalize_weights(new_weights)

    years = max(len(returns) / 12.0, 1e-9)
    cagr = value ** (1.0 / years) - 1.0
    return {
        "cagr": cagr,
        "turnover_per_year": total_turnover / years,
        "cumulative_cost_pctp": total_cost * 100.0,
        "ending_value": value,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--returns", default=None, help="monthly returns CSV with stocks,bonds,gold,cash")
    ap.add_argument("--cost-bps", type=float, default=15.0)
    args = ap.parse_args()

    returns = load_returns(Path(args.returns) if args.returns else None)
    rows = []
    labels = {"none": "무리밸런싱", "quarterly": "분기 캘린더", "band": "밴드"}
    results = {}
    for mode in ["none", "quarterly", "band"]:
        res = run_strategy(returns, mode, cost_bps=args.cost_bps)
        results[mode] = res
        rows.append({"policy": labels[mode], **res})

    table = pd.DataFrame(rows).set_index("policy")
    show = table.copy()
    show["cagr"] = show["cagr"] * 100
    print("=" * 72)
    print(f"Rebalancing cost check  (months={len(returns)}, cost={args.cost_bps:.1f}bps)")
    print("=" * 72)
    print(show[["cagr", "turnover_per_year", "cumulative_cost_pctp"]].round(3).to_string())

    band = results["band"]
    quarterly = results["quarterly"]
    verdict = "채택"
    reasons = []
    if band["turnover_per_year"] > quarterly["turnover_per_year"]:
        verdict = "보류"
        reasons.append("밴드 회전율이 분기 캘린더보다 큼")
    if band["cagr"] < quarterly["cagr"] - 0.01:
        verdict = "보류"
        reasons.append("밴드 CAGR이 분기 캘린더 대비 1%p 이상 낮음")
    if not reasons:
        reasons.append("밴드가 회전율을 낮추면서 성과 차이는 허용범위")

    print("-" * 72)
    print(f"판정: 밴드 방식 {verdict} - {'; '.join(reasons)}")
    return 0 if verdict == "채택" else 1


if __name__ == "__main__":
    raise SystemExit(main())



