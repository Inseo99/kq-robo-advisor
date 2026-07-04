"""Regime-shuffle validation for regime-based strategy selection.

Question
--------
Does a regime-conditional strategy selector add value versus fake shuffled
regimes and a static baseline?

This script uses only information available before each month:
  1. At month t, read the current regime label.
  2. Look back only at months < t with the same regime.
  3. Pick the candidate strategy with the best past same-regime Sharpe.
  4. Hold that strategy for month t.

Then it repeats the exact same process after shuffling regime labels. If the
real labels do not beat shuffled labels, the regime layer should be treated as a
risk-management/explanation layer rather than an alpha selector.

Usage:
    python tests\validation_regime_shuffle.py --data-dir tests --n 500
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

PERIODS_PER_YEAR = 12
DEFAULT_BENCHMARK = "KOSPI (벤치마크)"
DEFAULT_FALLBACK = "정적 60/40"
MISSING_REGIME_LABELS = {"", "nan", "NaN", "None", "NONE", "<NA>"}


def load_inputs(data_dir: Path) -> tuple[pd.DataFrame, pd.Series]:
    returns_path = data_dir / "strategy_monthly_returns.csv"
    regime_path = data_dir / "regime_labels_monthly.csv"
    if not returns_path.exists():
        raise FileNotFoundError(f"Missing strategy returns: {returns_path}")
    if not regime_path.exists():
        raise FileNotFoundError(f"Missing regime labels: {regime_path}")

    returns = pd.read_csv(returns_path, index_col=0, parse_dates=True).sort_index()
    regimes_df = pd.read_csv(regime_path, index_col=0, parse_dates=True).sort_index()
    regime_col = "regime" if "regime" in regimes_df.columns else regimes_df.columns[0]
    regime = regimes_df[regime_col].astype("string")
    regime = regime.mask(regime.isin(MISSING_REGIME_LABELS))

    returns.index = returns.index.to_period("M")
    regime.index = regime.index.to_period("M")
    common = returns.index.intersection(regime.index)
    return returns.loc[common], regime.loc[common].astype("string")


def choose_candidates(
    returns: pd.DataFrame,
    benchmark: str,
    fallback: str,
    include_regex: str | None,
    exclude_regex: str | None,
) -> list[str]:
    candidates = [col for col in returns.columns if col != benchmark]
    if include_regex:
        rx = re.compile(include_regex)
        candidates = [col for col in candidates if rx.search(col)]
    if exclude_regex:
        rx = re.compile(exclude_regex)
        candidates = [col for col in candidates if not rx.search(col)]
    if fallback in returns.columns and fallback not in candidates and fallback != benchmark:
        candidates.append(fallback)
    if not candidates:
        raise ValueError("No candidate strategy columns selected.")
    return candidates


def sharpe_score(values: pd.Series) -> float:
    clean = values.dropna()
    if len(clean) < 2:
        return float("nan")
    sd = clean.std(ddof=1)
    if not np.isfinite(sd) or sd == 0:
        return float("nan")
    return float(clean.mean() / sd * np.sqrt(PERIODS_PER_YEAR))


def performance_metrics(values: pd.Series) -> dict[str, float | int]:
    clean = values.dropna().astype(float)
    n = int(len(clean))
    if n == 0:
        return {
            "n_months": 0,
            "ann_return": np.nan,
            "ann_vol": np.nan,
            "sharpe": np.nan,
            "mdd": np.nan,
            "calmar": np.nan,
        }
    total = float((1.0 + clean).prod())
    ann_return = total ** (PERIODS_PER_YEAR / n) - 1.0 if total > 0 else np.nan
    sd = clean.std(ddof=1)
    ann_vol = float(sd * np.sqrt(PERIODS_PER_YEAR)) if n > 1 else np.nan
    sharpe = float(clean.mean() / sd * np.sqrt(PERIODS_PER_YEAR)) if n > 1 and sd > 0 else np.nan
    equity = (1.0 + clean).cumprod()
    mdd = float((equity / equity.cummax() - 1.0).min()) if n > 1 else 0.0
    calmar = float(ann_return / abs(mdd)) if np.isfinite(ann_return) and mdd < 0 else np.nan
    return {
        "n_months": n,
        "ann_return": float(ann_return) if np.isfinite(ann_return) else np.nan,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "mdd": mdd,
        "calmar": calmar,
    }


def walk_forward_regime_selector(
    returns: pd.DataFrame,
    regime: pd.Series,
    candidates: list[str],
    *,
    fallback: str,
    min_same_regime_months: int,
    metric: str,
) -> pd.DataFrame:
    """Fast walk-forward same-regime selector using cumulative past stats."""
    if metric != "sharpe":
        raise ValueError("Only metric='sharpe' is currently supported.")

    idx = returns.index
    candidate_cols = [col for col in candidates if col in returns.columns]
    if not candidate_cols:
        raise ValueError("No candidate columns exist in returns DataFrame.")

    values = returns[candidate_cols].to_numpy(dtype=float)
    fallback_idx = candidate_cols.index(fallback) if fallback in candidate_cols else 0
    strategy_names = np.array(candidate_cols, dtype=object)

    valid_regimes = sorted(str(x) for x in regime.dropna().unique())
    regime_to_i = {name: i for i, name in enumerate(valid_regimes)}
    n_regimes = max(len(valid_regimes), 1)
    n_strategies = len(candidate_cols)

    counts = np.zeros((n_regimes, n_strategies), dtype=float)
    sums = np.zeros((n_regimes, n_strategies), dtype=float)
    sumsq = np.zeros((n_regimes, n_strategies), dtype=float)

    rows: list[dict] = []
    for i, timestamp in enumerate(idx):
        label = regime.iloc[i]
        label_key = None if pd.isna(label) else str(label)
        r_i = regime_to_i.get(label_key)

        selected_idx = fallback_idx
        reason = "fallback"

        if r_i is not None:
            eligible = counts[r_i] >= float(min_same_regime_months)
            if eligible.any():
                mean = np.divide(sums[r_i], counts[r_i], out=np.zeros(n_strategies), where=counts[r_i] > 0)
                variance_num = sumsq[r_i] - (sums[r_i] ** 2) / np.maximum(counts[r_i], 1.0)
                variance = np.divide(
                    variance_num,
                    counts[r_i] - 1.0,
                    out=np.full(n_strategies, np.nan),
                    where=counts[r_i] > 1.0,
                )
                sd = np.sqrt(np.maximum(variance, 0.0))
                scores = np.divide(
                    mean,
                    sd,
                    out=np.full(n_strategies, np.nan),
                    where=sd > 0,
                ) * np.sqrt(PERIODS_PER_YEAR)
                scores[~eligible] = np.nan
                if np.isfinite(scores).any():
                    selected_idx = int(np.nanargmax(scores))
                    reason = "same_regime_history"

        realized = values[i, selected_idx]
        if not np.isfinite(realized):
            realized = values[i, fallback_idx]
            selected_idx = fallback_idx
            reason = "fallback_missing_selected"

        rows.append(
            {
                "date": timestamp.to_timestamp("M"),
                "regime": label,
                "selected_strategy": strategy_names[selected_idx],
                "selection_reason": reason,
                "return": realized,
            }
        )

        # Update after month t selection so t returns never affect t decision.
        if r_i is not None:
            month_values = values[i]
            valid = np.isfinite(month_values)
            counts[r_i, valid] += 1.0
            sums[r_i, valid] += month_values[valid]
            sumsq[r_i, valid] += month_values[valid] ** 2

    return pd.DataFrame(rows).set_index("date")

def shuffled_regime(regime: pd.Series, rng: np.random.Generator) -> pd.Series:
    out = regime.copy()
    valid = out.dropna()
    shuffled = valid.to_numpy(copy=True)
    rng.shuffle(shuffled)
    out.loc[valid.index] = shuffled
    return out


def run_placebo(
    returns: pd.DataFrame,
    regime: pd.Series,
    candidates: list[str],
    *,
    fallback: str,
    min_same_regime_months: int,
    n: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    actual_path = walk_forward_regime_selector(
        returns,
        regime,
        candidates,
        fallback=fallback,
        min_same_regime_months=min_same_regime_months,
        metric="sharpe",
    )
    actual_metrics = performance_metrics(actual_path["return"])

    rng = np.random.default_rng(seed)
    placebo_rows = []
    for i in range(n):
        fake_regime = shuffled_regime(regime, rng)
        fake_path = walk_forward_regime_selector(
            returns,
            fake_regime,
            candidates,
            fallback=fallback,
            min_same_regime_months=min_same_regime_months,
            metric="sharpe",
        )
        row = performance_metrics(fake_path["return"])
        row["iteration"] = i + 1
        placebo_rows.append(row)
    placebo = pd.DataFrame(placebo_rows)

    baseline = returns[fallback] if fallback in returns.columns else pd.Series(dtype=float)
    benchmark = returns[DEFAULT_BENCHMARK] if DEFAULT_BENCHMARK in returns.columns else pd.Series(dtype=float)

    summary_rows = [
        {"portfolio": "real_regime_selector", **actual_metrics},
        {"portfolio": f"fallback:{fallback}", **performance_metrics(baseline)},
    ]
    if not benchmark.empty:
        summary_rows.append({"portfolio": f"benchmark:{DEFAULT_BENCHMARK}", **performance_metrics(benchmark)})

    summary = pd.DataFrame(summary_rows)
    for metric in ["sharpe", "calmar", "mdd"]:
        actual_value = float(actual_metrics.get(metric, np.nan))
        if not np.isfinite(actual_value) or placebo.empty:
            summary.loc[summary["portfolio"] == "real_regime_selector", f"placebo_p_{metric}"] = np.nan
            summary.loc[summary["portfolio"] == "real_regime_selector", f"placebo_pct_{metric}"] = np.nan
            continue
        if metric == "mdd":
            better_or_equal = placebo[metric] >= actual_value
            percentile = float((placebo[metric] <= actual_value).mean() * 100.0)
        else:
            better_or_equal = placebo[metric] >= actual_value
            percentile = float((placebo[metric] <= actual_value).mean() * 100.0)
        p_value = float((1 + better_or_equal.sum()) / (len(placebo) + 1))
        mask = summary["portfolio"] == "real_regime_selector"
        summary.loc[mask, f"placebo_p_{metric}"] = p_value
        summary.loc[mask, f"placebo_pct_{metric}"] = percentile

    return summary, placebo, actual_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="tests")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--benchmark", default=DEFAULT_BENCHMARK)
    parser.add_argument("--fallback", default=DEFAULT_FALLBACK)
    parser.add_argument("--min-same-regime-months", type=int, default=6)
    parser.add_argument("--include-regex", default=None)
    parser.add_argument("--exclude-regex", default=r"^(quant|quant_s2|robo)$")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir) if args.out_dir else data_dir / "analysis_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    returns, regime = load_inputs(data_dir)
    candidates = choose_candidates(
        returns,
        benchmark=args.benchmark,
        fallback=args.fallback,
        include_regex=args.include_regex,
        exclude_regex=args.exclude_regex,
    )

    summary, placebo, path = run_placebo(
        returns,
        regime,
        candidates,
        fallback=args.fallback,
        min_same_regime_months=args.min_same_regime_months,
        n=args.n,
        seed=args.seed,
    )

    summary_path = out_dir / "regime_shuffle_summary.csv"
    placebo_path = out_dir / "regime_shuffle_placebo.csv"
    path_path = out_dir / "regime_shuffle_actual_path.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    placebo.to_csv(placebo_path, index=False, encoding="utf-8-sig")
    path.to_csv(path_path, encoding="utf-8-sig")

    print("=" * 72)
    print("Regime Shuffle Validation")
    print("=" * 72)
    print(f"입력 폴더: {data_dir}")
    print(f"후보 전략 수: {len(candidates)}")
    print(f"후보 전략: {', '.join(candidates)}")
    print(f"셔플 횟수: {args.n}")
    print("-" * 72)
    print(summary.round(4).to_string(index=False))

    actual = summary[summary["portfolio"] == "real_regime_selector"].iloc[0]
    p_sharpe = actual.get("placebo_p_sharpe", np.nan)
    p_calmar = actual.get("placebo_p_calmar", np.nan)
    if np.isfinite(p_sharpe) and p_sharpe < 0.05:
        print("\n판정: Sharpe 기준 실제 국면 선택기가 셔플 국면 대비 유의합니다.")
    elif np.isfinite(p_calmar) and p_calmar < 0.05:
        print("\n판정: Calmar 기준 실제 국면 선택기가 셔플 국면 대비 유의합니다.")
    else:
        print("\n판정: 셔플 국면 대비 유의성이 약합니다. 국면은 알파 엔진보다 위험관리 레이어로 해석하세요.")

    print(f"\n결과 저장: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


