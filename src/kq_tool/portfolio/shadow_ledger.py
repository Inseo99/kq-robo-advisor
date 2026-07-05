"""Shadow ledger for ongoing rebalancing-policy self-checks.

The adopted policy is not trusted forever. This module reconstructs, from the
same monthly asset returns, three parallel policy paths:

- band: adopted band-triggered policy
- quarterly: quarterly calendar full rebalance control
- never: no-rebalance control

The ledger is deterministic and idempotent: every run rebuilds the full history
from `monthly_returns.csv`. That makes the result auditable and enables vintage
invariance tests; no hidden state file can drift away from the data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from .rebalancing import RebalancePolicy, decide, normalize_weights, turnover

MODES = ("band", "quarterly", "never")
DEFAULT_TARGET = {
    "stocks": 0.40,
    "bonds": 0.35,
    "gold": 0.15,
    "cash": 0.10,
}
DEFAULT_COST_BPS = 15.0
CAGR_REVIEW_TOLERANCE = 0.002


def _target_dict(target: Mapping[str, float] | pd.Series | None = None) -> dict[str, float]:
    return normalize_weights(dict(target if target is not None else DEFAULT_TARGET))


def _prepare_returns(returns: pd.DataFrame, target: Mapping[str, float]) -> pd.DataFrame:
    frame = returns.copy()
    if "date" in frame.columns:
        frame["date"] = pd.to_datetime(frame["date"])
        frame = frame.set_index("date")
    frame.index = pd.to_datetime(frame.index).to_period("M").to_timestamp("M")
    missing = [asset for asset in target if asset not in frame.columns]
    if missing:
        raise ValueError(f"missing required return columns: {missing}")
    return frame[list(target)].sort_index().dropna()


def _apply_month(weights: dict[str, float], returns: pd.Series) -> tuple[dict[str, float], float]:
    gross_weights = {
        asset: float(weights.get(asset, 0.0)) * (1.0 + float(returns[asset]))
        for asset in weights
    }
    portfolio_return = sum(gross_weights.values()) - 1.0
    return normalize_weights(gross_weights), portfolio_return


def build_ledger(
    returns: pd.DataFrame,
    target: Mapping[str, float] | pd.Series | None = None,
    *,
    cost_bps: float = DEFAULT_COST_BPS,
    policy: RebalancePolicy | None = None,
) -> pd.DataFrame:
    """Build a monthly shadow ledger for band/quarterly/never policies."""

    target_weights = _target_dict(target)
    returns = _prepare_returns(returns, target_weights)
    policy = policy or RebalancePolicy()
    cost_rate = max(0.0, float(cost_bps)) / 10_000.0

    state = {
        mode: {"weights": dict(target_weights), "nav": 1.0}
        for mode in MODES
    }
    rows: list[dict[str, object]] = []

    for month_number, (date, row_returns) in enumerate(returns.iterrows(), start=1):
        row: dict[str, object] = {"date": date}
        for mode in MODES:
            mode_state = state[mode]
            drifted_weights, month_return = _apply_month(mode_state["weights"], row_returns)
            nav_before_cost = float(mode_state["nav"]) * (1.0 + month_return)

            action = "hold"
            trigger = "NO_TRADE"
            new_weights = drifted_weights
            trade_turnover = 0.0

            if mode == "quarterly":
                if month_number % 3 == 0:
                    action = "trade"
                    trigger = "QUARTERLY_CALENDAR"
                    new_weights = dict(target_weights)
                    trade_turnover = turnover(drifted_weights, new_weights)
            elif mode == "band":
                decision = decide(drifted_weights, target_weights, policy=policy)
                action = decision.action
                trigger = decision.trigger
                new_weights = decision.target_weights
                trade_turnover = decision.turnover
            elif mode == "never":
                pass
            else:  # pragma: no cover - MODES is fixed
                raise ValueError(f"unknown mode: {mode!r}")

            cost = trade_turnover * cost_rate
            nav_after_cost = nav_before_cost * (1.0 - cost)
            mode_state["weights"] = normalize_weights(new_weights)
            mode_state["nav"] = nav_after_cost

            row[f"{mode}_nav"] = nav_after_cost
            row[f"{mode}_return"] = month_return
            row[f"{mode}_turnover"] = trade_turnover
            row[f"{mode}_cost"] = cost
            row[f"{mode}_action"] = action
            row[f"{mode}_trigger"] = trigger
            for asset, weight in mode_state["weights"].items():
                row[f"{mode}_w_{asset}"] = weight
        rows.append(row)

    ledger = pd.DataFrame(rows).set_index("date")
    ledger.attrs["cost_bps"] = float(cost_bps)
    ledger.attrs["target"] = target_weights
    return ledger


def _policy_summary(ledger: pd.DataFrame, mode: str) -> dict[str, float | int]:
    n_months = len(ledger)
    nav = ledger[f"{mode}_nav"]
    ending_nav = float(nav.iloc[-1])
    cagr = ending_nav ** (12.0 / max(n_months, 1)) - 1.0
    return {
        "nav": round(ending_nav, 4),
        "cagr": round(cagr, 4),
        "turnover_yr": round(float(ledger[f"{mode}_turnover"].sum()) / n_months * 12.0, 4),
        "cum_cost_pctp": round(float(ledger[f"{mode}_cost"].sum()) * 100.0, 4),
        "trade_months": int((ledger[f"{mode}_turnover"] > 0).sum()),
    }


def ledger_summary(ledger: pd.DataFrame) -> dict[str, object]:
    """Summarize the ledger for UI payloads and policy-review triggers."""

    if ledger.empty:
        raise ValueError("ledger is empty")

    out: dict[str, object] = {
        "months": int(len(ledger)),
        "period": f"{ledger.index[0]:%Y-%m} ~ {ledger.index[-1]:%Y-%m}",
        "cost_bps_oneway": float(ledger.attrs.get("cost_bps", np.nan)),
        "target_weights": ledger.attrs.get("target", DEFAULT_TARGET),
    }
    for mode in MODES:
        out[mode] = _policy_summary(ledger, mode)

    band = out["band"]  # type: ignore[assignment]
    quarterly = out["quarterly"]  # type: ignore[assignment]
    never = out["never"]  # type: ignore[assignment]
    cagr_gap = float(band["cagr"]) - float(quarterly["cagr"])  # type: ignore[index]
    cost_saved = float(quarterly["cum_cost_pctp"]) - float(band["cum_cost_pctp"])  # type: ignore[index]
    turnover_saved = float(quarterly["turnover_yr"]) - float(band["turnover_yr"])  # type: ignore[index]

    if cagr_gap >= -CAGR_REVIEW_TOLERANCE:
        verdict = "밴드 유지"
    else:
        verdict = "정책 재검토 트리거 - 분기 캘린더가 허용오차 밖에서 우월"

    out["band_vs_quarterly"] = {
        "cost_saved_pctp": round(cost_saved, 4),
        "turnover_saved_yr": round(turnover_saved, 4),
        "cagr_gap_pctp": round(cagr_gap * 100.0, 4),
        "tolerance_pctp": round(CAGR_REVIEW_TOLERANCE * 100.0, 4),
        "verdict": verdict,
    }
    out["band_vs_never"] = {
        "cagr_gap_pctp": round((float(band["cagr"]) - float(never["cagr"])) * 100.0, 4),  # type: ignore[index]
        "turnover_added_yr": round(float(band["turnover_yr"]) - float(never["turnover_yr"]), 4),  # type: ignore[index]
    }
    return out


def run_and_save(
    returns_csv: Path,
    out_dir: Path,
    *,
    cost_bps: float = DEFAULT_COST_BPS,
    target: Mapping[str, float] | pd.Series | None = None,
) -> dict[str, object]:
    """Build the ledger from CSV and save CSV + JSON state artifacts."""

    returns = pd.read_csv(returns_csv, index_col=0, parse_dates=True)
    ledger = build_ledger(returns, target=target, cost_bps=cost_bps)
    out_dir.mkdir(parents=True, exist_ok=True)
    ledger.round(8).to_csv(out_dir / "shadow_ledger.csv", encoding="utf-8-sig")
    summary = ledger_summary(ledger)
    (out_dir / "shadow_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--returns", default="data/assets/monthly_returns.csv")
    parser.add_argument("--out-dir", default="data/state")
    parser.add_argument("--cost-bps", type=float, default=DEFAULT_COST_BPS)
    args = parser.parse_args(argv)

    root = Path(__file__).resolve().parents[3]
    returns_csv = root / args.returns
    if not returns_csv.exists():
        print(f"[stop] {returns_csv} 없음")
        return 1
    summary = run_and_save(returns_csv, root / args.out_dir, cost_bps=args.cost_bps)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\n저장: data/state/shadow_ledger.csv, data/state/shadow_summary.json")
    return 0


__all__ = [
    "CAGR_REVIEW_TOLERANCE",
    "DEFAULT_COST_BPS",
    "DEFAULT_TARGET",
    "MODES",
    "build_ledger",
    "ledger_summary",
    "run_and_save",
]


if __name__ == "__main__":
    raise SystemExit(main())
