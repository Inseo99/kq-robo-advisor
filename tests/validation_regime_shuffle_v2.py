"""Regime shuffle validation v2: separate signal value from selector value.

The previous shuffle test mixed two questions:
1. Does the regime label itself contain useful allocation information?
2. Does a strategy selector built on top of regimes improve performance?

This v2 separates them.

Tier 1 (default): fixed pre-registered regime -> asset weights mapping.
    Question: Do regime labels carry useful allocation information?
Tier 2 (--tier2): walk-forward selector over a small candidate set.
    Question: Does an extra selector improve on the fixed mapping?

Comparison ladder:
    (a) static 60/40
    (b) defense-matched static portfolio, using the strategy's average weights
    (c) block-shuffled regime placebo, preserving regime run lengths
    (d) actual regime strategy

Verdict requires (d) beating (a), beating (b), and being significant versus (c).

Usage:
    python tests\validation_regime_shuffle_v2.py --n 500 --seed 42
    python tests\validation_regime_shuffle_v2.py --n 500 --tier2

Real data inputs:
    --returns data/assets/monthly_returns.csv   columns: date,stocks,bonds,gold,cash
    --labels  data/macro/regime_labels.csv      columns: date,regime

If real data is missing, the script runs a synthetic smoke test. Do not use the
synthetic p-values as project evidence.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kq_tool.regime.regime_labels import REGIMES  # noqa: E402

OUT_DIR = ROOT / "tests" / "analysis_outputs"
PERIODS_PER_YEAR = 12
ASSETS = ["stocks", "bonds", "gold", "cash"]
COST_BPS_ONEWAY = 15.0

# Pre-registered mapping. Do not tune from performance without recording the
# number of attempts in DSR / multiple-testing correction.
FIXED_MAP = {
    "골디락스": {"stocks": 0.70, "bonds": 0.20, "gold": 0.05, "cash": 0.05},
    "리플레이션": {"stocks": 0.50, "bonds": 0.15, "gold": 0.25, "cash": 0.10},
    "스태그플레이션": {"stocks": 0.20, "bonds": 0.15, "gold": 0.30, "cash": 0.35},
    "디플레이션": {"stocks": 0.15, "bonds": 0.60, "gold": 0.05, "cash": 0.20},
}

TIER2_CANDIDATES = {
    "고정매핑": None,
    "동일비중": {regime: {asset: 0.25 for asset in ASSETS} for regime in REGIMES},
    "주식중심": {
        regime: {"stocks": 0.80, "bonds": 0.10, "gold": 0.05, "cash": 0.05}
        for regime in REGIMES
    },
    "채권중심": {
        regime: {"stocks": 0.20, "bonds": 0.60, "gold": 0.10, "cash": 0.10}
        for regime in REGIMES
    },
    "방어형": {
        regime: {"stocks": 0.10, "bonds": 0.30, "gold": 0.30, "cash": 0.30}
        for regime in REGIMES
    },
}


def _normalize_month_index(obj: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    out = obj.copy()
    out.index = pd.to_datetime(out.index).to_period("M").to_timestamp("M")
    return out.sort_index()


def validate_returns(returns: pd.DataFrame) -> pd.DataFrame:
    missing = [asset for asset in ASSETS if asset not in returns.columns]
    if missing:
        raise KeyError(f"Asset return file must contain {ASSETS}; missing={missing}")
    out = returns[ASSETS].apply(pd.to_numeric, errors="coerce")
    return _normalize_month_index(out)


def weights_from_labels(labels: pd.Series, mapping: dict[str, dict[str, float]]) -> pd.DataFrame:
    weights = pd.DataFrame(index=labels.index, columns=ASSETS, dtype=float)
    for timestamp, label in labels.items():
        if pd.isna(label):
            continue
        label = str(label)
        if label not in mapping:
            continue
        weights.loc[timestamp] = pd.Series(mapping[label])
    return weights.dropna(how="any")


def run_portfolio(
    weights: pd.DataFrame,
    returns: pd.DataFrame,
    cost_bps: float = COST_BPS_ONEWAY,
) -> pd.DataFrame:
    """Use weights decided at month t for month t+1 returns."""

    delayed = weights.shift(1).dropna(how="any")
    common = delayed.index.intersection(returns.index)
    if common.empty:
        return pd.DataFrame(columns=["gross", "net", "turnover", "eq_w"])

    weights_i = delayed.loc[common, ASSETS]
    returns_i = returns.loc[common, ASSETS]
    gross = (weights_i * returns_i).sum(axis=1)
    turnover = weights_i.diff().abs().sum(axis=1).fillna(0.0)
    net = gross - turnover * cost_bps / 1e4
    return pd.DataFrame(
        {"gross": gross, "net": net, "turnover": turnover, "eq_w": weights_i["stocks"]}
    )


def metrics(net: pd.Series) -> dict[str, float | int]:
    clean = pd.to_numeric(net, errors="coerce").dropna()
    n = int(len(clean))
    if n == 0:
        return {"n_months": 0, "CAGR": np.nan, "Sharpe": np.nan, "MDD": np.nan, "Calmar": np.nan}

    equity = (1.0 + clean).cumprod()
    ending = float(equity.iloc[-1])
    cagr = ending ** (PERIODS_PER_YEAR / n) - 1.0 if ending > 0 else np.nan
    vol = float(clean.std(ddof=1) * np.sqrt(PERIODS_PER_YEAR)) if n > 1 else np.nan
    sharpe = float(clean.mean() * PERIODS_PER_YEAR / vol) if np.isfinite(vol) and vol > 0 else np.nan
    mdd = float((equity / equity.cummax() - 1.0).min()) if n > 1 else 0.0
    calmar = float(cagr / abs(mdd)) if np.isfinite(cagr) and mdd < 0 else np.nan
    return {"n_months": n, "CAGR": cagr, "Sharpe": sharpe, "MDD": mdd, "Calmar": calmar}


def block_shuffle_labels(labels: pd.Series, rng: np.random.Generator) -> pd.Series:
    """Shuffle contiguous regime runs, preserving duration structure."""

    clean = labels.dropna()
    runs: list[tuple[str, int]] = []
    for label in clean:
        label = str(label)
        if runs and runs[-1][0] == label:
            runs[-1] = (label, runs[-1][1] + 1)
        else:
            runs.append((label, 1))

    order = rng.permutation(len(runs))
    shuffled: list[str] = []
    for run_i in order:
        label, length = runs[int(run_i)]
        shuffled.extend([label] * length)
    return pd.Series(shuffled[: len(clean)], index=clean.index, name=labels.name)


def tier2_selector(labels: pd.Series, returns: pd.DataFrame, lookback_m: int = 60) -> pd.DataFrame:
    """Pick the best candidate for the current regime using only past data."""

    weights = pd.DataFrame(index=labels.index, columns=ASSETS, dtype=float)
    for i, timestamp in enumerate(labels.index):
        label = labels.loc[timestamp]
        if pd.isna(label) or i < lookback_m:
            continue
        label = str(label)

        past_idx = labels.index[max(0, i - lookback_m): i]
        same_regime = past_idx[labels.loc[past_idx] == label]
        best_name = "고정매핑"
        best_sharpe = -np.inf
        for name, mapping in TIER2_CANDIDATES.items():
            active_mapping = FIXED_MAP if mapping is None else mapping
            if label not in active_mapping:
                continue
            w = pd.Series(active_mapping[label])
            history = (returns.loc[same_regime, ASSETS] * w).sum(axis=1).dropna()
            if len(history) < 6:
                continue
            sd = history.std(ddof=1)
            score = history.mean() / sd if sd > 0 else -np.inf
            if score > best_sharpe:
                best_name = name
                best_sharpe = float(score)
        active_mapping = FIXED_MAP if TIER2_CANDIDATES[best_name] is None else TIER2_CANDIDATES[best_name]
        weights.loc[timestamp] = pd.Series(active_mapping[label])
    return weights.dropna(how="any")


def load_real_data(returns_path: Path, labels_path: Path) -> tuple[pd.DataFrame, pd.Series] | None:
    if not returns_path.exists() or not labels_path.exists():
        return None
    returns = pd.read_csv(returns_path, parse_dates=["date"]).set_index("date")
    labels_df = pd.read_csv(labels_path, parse_dates=["date"]).set_index("date")
    regime_col = "regime" if "regime" in labels_df.columns else labels_df.columns[0]
    labels = labels_df[regime_col].astype("string")
    return validate_returns(returns), _normalize_month_index(labels)


def synthetic_data(seed: int) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    n = 240
    idx = pd.date_range("2006-01-31", periods=n, freq="ME")

    state = np.zeros(n, dtype=int)
    for i in range(1, n):
        state[i] = state[i - 1] if rng.random() < 0.93 else int(rng.integers(0, 4))

    monthly_mu = {
        0: (0.012, 0.002, 0.001, 0.002),
        1: (0.008, -0.001, 0.006, 0.002),
        2: (-0.008, 0.000, 0.008, 0.002),
        3: (-0.004, 0.006, 0.001, 0.002),
    }
    returns = pd.DataFrame(index=idx, columns=ASSETS, dtype=float)
    for i in range(n):
        mu = monthly_mu[state[i]]
        returns.iloc[i] = [
            rng.normal(mu[0], 0.045),
            rng.normal(mu[1], 0.012),
            rng.normal(mu[2], 0.035),
            rng.normal(mu[3], 0.001),
        ]

    labels = pd.Series([REGIMES[s] for s in state], index=idx, name="regime").shift(2)
    return returns, labels


def p_value(placebo: pd.DataFrame, metric: str, actual: float) -> float:
    if not np.isfinite(actual) or metric not in placebo:
        return np.nan
    # For MDD, values closer to zero are better, so higher is still better.
    return float((placebo[metric] >= actual).mean())


def run_tier1(returns: pd.DataFrame, labels: pd.Series, n: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)

    real_weights = weights_from_labels(labels, FIXED_MAP)
    real_path = run_portfolio(real_weights, returns)
    real_metrics = metrics(real_path["net"])

    static_6040 = pd.DataFrame(
        {"stocks": 0.60, "bonds": 0.40, "gold": 0.0, "cash": 0.0},
        index=labels.index,
    )
    static_6040_metrics = metrics(run_portfolio(static_6040, returns)["net"])

    avg_weights = real_weights.mean()
    matched = pd.DataFrame([avg_weights] * len(labels), index=labels.index)
    matched_metrics = metrics(run_portfolio(matched, returns)["net"])

    placebo_rows = []
    for _ in range(n):
        fake_labels = block_shuffle_labels(labels, rng)
        fake_weights = weights_from_labels(fake_labels, FIXED_MAP)
        placebo_rows.append(metrics(run_portfolio(fake_weights, returns)["net"]))
    placebo = pd.DataFrame(placebo_rows)

    turnover_yr = float(real_path["turnover"].mean() * PERIODS_PER_YEAR) if not real_path.empty else np.nan
    summary = pd.DataFrame(
        [
            {"name": "static_6040", **static_6040_metrics},
            {"name": "defense_matched_static", **matched_metrics, **{f"avg_{a}": avg_weights[a] for a in ASSETS}},
            {"name": "real_regime_fixed_map", **real_metrics, "turnover_yr": turnover_yr},
        ]
    )
    for metric in ["Sharpe", "Calmar", "MDD"]:
        summary.loc[summary["name"] == "real_regime_fixed_map", f"placebo_p_{metric}"] = p_value(
            placebo, metric, float(real_metrics.get(metric, np.nan))
        )
    return summary, placebo, real_path


def print_summary(summary: pd.DataFrame, placebo: pd.DataFrame) -> None:
    row_map = {row["name"]: row for _, row in summary.iterrows()}
    print(f"\n{'':<24}{'CAGR':>9}{'Sharpe':>9}{'MDD':>9}{'Calmar':>9}{'Turnover':>10}")
    for name, display in [
        ("static_6040", "(a) 정적 60/40"),
        ("defense_matched_static", "(b) 방어매칭 정적"),
    ]:
        row = row_map[name]
        print(f"{display:<24}{row['CAGR']:>8.2%}{row['Sharpe']:>9.3f}{row['MDD']:>8.2%}{row['Calmar']:>9.3f}")
    pmean = placebo.mean(numeric_only=True)
    print(f"{'(c) 셔플 평균':<24}{pmean['CAGR']:>8.2%}{pmean['Sharpe']:>9.3f}{pmean['MDD']:>8.2%}{pmean['Calmar']:>9.3f}")
    row = row_map["real_regime_fixed_map"]
    print(
        f"{'(d) 실제 국면 전략':<24}{row['CAGR']:>8.2%}{row['Sharpe']:>9.3f}"
        f"{row['MDD']:>8.2%}{row['Calmar']:>9.3f}{row['turnover_yr']:>9.2f}x"
    )

    print("\n셔플 p-value, 낮을수록 좋음:")
    print(
        f"  Sharpe p = {row['placebo_p_Sharpe']:.4f}"
        f"   Calmar p = {row['placebo_p_Calmar']:.4f}"
        f"   MDD p = {row['placebo_p_MDD']:.4f}"
    )

    beats_static = row["Sharpe"] > row_map["static_6040"]["Sharpe"]
    beats_matched = row["Sharpe"] > row_map["defense_matched_static"]["Sharpe"]
    significant = row["placebo_p_Sharpe"] < 0.10
    verdict = "국면에 배분 정보 있음" if (beats_static and beats_matched and significant) else "국면 정보 불충분 - 위험관리/설명 보조 레이어로 한정"
    print("\n판정 사다리:")
    print(f"  (d) > (a) 60/40      : {'예' if beats_static else '아니오'}")
    print(f"  (d) > (b) 방어매칭   : {'예' if beats_matched else '아니오'}")
    print(f"  셔플 대비 유의(p<.1) : {'예' if significant else '아니오'}")
    print(f"  종합: {verdict}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=500, help="number of block-shuffle placebo runs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tier2", action="store_true", help="also run Tier 2 selector over registered candidates")
    parser.add_argument("--returns", default="data/assets/monthly_returns.csv")
    parser.add_argument("--labels", default="data/macro/regime_labels.csv")
    args = parser.parse_args()

    real = load_real_data(Path(args.returns), Path(args.labels))
    is_synth = real is None
    if real is None:
        print("[smoke mode] Real data not found. Running synthetic pipeline test only.")
        print("             Do not use synthetic p-values as project evidence.\n")
        returns, labels = synthetic_data(args.seed)
    else:
        returns, labels = real

    labels = labels.dropna()
    common = labels.index.intersection(returns.index)
    labels = labels.loc[common]
    returns = returns.loc[common]

    print("=" * 72)
    print("Regime Shuffle Validation v2 - Tier 1 fixed mapping" + (" + Tier 2" if args.tier2 else ""))
    print("=" * 72)
    print(f"months={len(common)}, placebo_n={args.n}, synthetic={is_synth}")

    summary, placebo, actual_path = run_tier1(returns, labels, args.n, args.seed)
    print_summary(summary, placebo)

    if args.tier2:
        t2_weights = tier2_selector(labels, returns)
        t2_metrics = metrics(run_portfolio(t2_weights, returns)["net"])
        tier1_sharpe = float(summary.loc[summary["name"] == "real_regime_fixed_map", "Sharpe"].iloc[0])
        print(f"\nTier 2 selector, candidates={len(TIER2_CANDIDATES)} - record as DSR trials:")
        print(f"  Tier2 Sharpe {t2_metrics['Sharpe']:.3f} vs Tier1 {tier1_sharpe:.3f}")
        print("  " + ("선택기 기여 있음" if t2_metrics["Sharpe"] > tier1_sharpe else "Tier 1 유지, 선택기 보류"))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT_DIR / "regime_shuffle_v2_summary.csv", index=False, encoding="utf-8-sig")
    placebo.to_csv(OUT_DIR / "regime_shuffle_v2_placebo.csv", index=False, encoding="utf-8-sig")
    path = actual_path.copy()
    path["regime"] = labels.reindex(path.index)
    path.to_csv(OUT_DIR / "regime_shuffle_v2_actual_path.csv", encoding="utf-8-sig")
    print(f"\n결과 저장: {OUT_DIR / 'regime_shuffle_v2_*.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
