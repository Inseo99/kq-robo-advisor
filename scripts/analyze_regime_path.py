r"""Analyze attribution of the realized regime allocation path.

Reads ``tests/analysis_outputs/regime_shuffle_v2_actual_path.csv`` from the
fixed-map shuffle test and explains where the strategy lost versus static 60/40.

Outputs are saved to ``tests/analysis_outputs``:
  - regime_path_attribution_by_regime.csv
  - regime_path_attribution_by_year.csv
  - regime_path_worst_months.csv
  - regime_path_verdict.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "tests" / "analysis_outputs"
ASSETS = ["stocks", "bonds", "gold", "cash"]
FIXED_MAP = {
    "골디락스": {"stocks": 0.70, "bonds": 0.20, "gold": 0.05, "cash": 0.05},
    "리플레이션": {"stocks": 0.50, "bonds": 0.15, "gold": 0.25, "cash": 0.10},
    "스태그플레이션": {"stocks": 0.20, "bonds": 0.15, "gold": 0.30, "cash": 0.35},
    "디플레이션": {"stocks": 0.15, "bonds": 0.60, "gold": 0.05, "cash": 0.20},
}


def _month_index(obj: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    out = obj.copy()
    out.index = pd.to_datetime(out.index).to_period("M").to_timestamp("M")
    return out.sort_index()


def load_inputs(actual_path: Path, returns_path: Path, labels_path: Path) -> pd.DataFrame:
    path = pd.read_csv(actual_path, parse_dates=["date"]).set_index("date")
    returns = pd.read_csv(returns_path, index_col=0, parse_dates=True)
    labels = pd.read_csv(labels_path, parse_dates=["date"]).set_index("date")
    regime_col = "regime" if "regime" in labels.columns else labels.columns[0]

    path = _month_index(path)
    returns = _month_index(returns[ASSETS].apply(pd.to_numeric, errors="coerce"))
    labels_s = _month_index(labels[regime_col].astype("string"))

    common = path.index.intersection(returns.index)
    df = path.loc[common].copy()
    rets = returns.loc[common]
    df[ASSETS] = rets[ASSETS]
    df["static_6040"] = 0.60 * df["stocks"] + 0.40 * df["bonds"]
    df["actual"] = pd.to_numeric(df["net"], errors="coerce")
    df["gap_vs_6040"] = df["actual"] - df["static_6040"]

    # run_portfolio uses weights decided at month t-1 for returns at month t.
    decision_regime = labels_s.shift(1).reindex(df.index)
    df["decision_regime"] = decision_regime.astype("string")
    df["label_regime"] = labels_s.reindex(df.index).astype("string")
    return df.dropna(subset=["actual", "static_6040", "decision_regime"])


def compound(series: pd.Series) -> float:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return np.nan
    return float((1.0 + clean).prod() - 1.0)


def summarize_by_regime(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for regime, group in df.groupby("decision_regime", observed=True):
        rows.append(
            {
                "decision_regime": regime,
                "months": int(len(group)),
                "avg_eq_w": float(group["eq_w"].mean()),
                "actual_compound": compound(group["actual"]),
                "static_6040_compound": compound(group["static_6040"]),
                "gap_compound": compound(group["actual"]) - compound(group["static_6040"]),
                "gap_sum_monthly": float(group["gap_vs_6040"].sum()),
                "actual_mean_m": float(group["actual"].mean()),
                "stocks_mean_m": float(group["stocks"].mean()),
                "bonds_mean_m": float(group["bonds"].mean()),
                "gold_mean_m": float(group["gold"].mean()),
                "cash_mean_m": float(group["cash"].mean()),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values("gap_sum_monthly")


def summarize_by_year(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for year, group in df.groupby(df.index.year):
        actual = compound(group["actual"])
        static = compound(group["static_6040"])
        rows.append(
            {
                "year": int(year),
                "months": int(len(group)),
                "actual": actual,
                "static_6040": static,
                "gap_actual_minus_6040": actual - static,
                "stocks": compound(group["stocks"]),
                "bonds": compound(group["bonds"]),
                "gold": compound(group["gold"]),
                "cash": compound(group["cash"]),
                "dominant_decision_regime": group["decision_regime"].mode().iloc[0],
                "avg_eq_w": float(group["eq_w"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("year")


def concentration_verdict(by_year: pd.DataFrame) -> tuple[str, float, str]:
    losses = by_year.copy()
    losses["loss_to_6040"] = (-losses["gap_actual_minus_6040"]).clip(lower=0.0)
    total_loss = float(losses["loss_to_6040"].sum())
    if total_loss <= 0:
        return "실제 국면 전략이 60/40 대비 손실 집중을 보이지 않았습니다.", 0.0, "none"

    worst = losses.sort_values("loss_to_6040", ascending=False).head(2)
    top_share = float(worst["loss_to_6040"].sum() / total_loss)
    years = ", ".join(str(int(y)) for y in worst["year"])
    if top_share >= 0.60:
        verdict = (
            f"손실 갭의 {top_share:.1%}가 상위 2개 연도({years})에 집중되어 있습니다. "
            "결론 문구는 '국면 자체가 무정보'보다 '최근 강세장 표본에서 방어적 고정 매핑의 "
            "기회비용이 컸다'가 더 정확합니다."
        )
        tag = "concentrated"
    else:
        verdict = (
            f"손실 갭의 상위 2개 연도 집중도가 {top_share:.1%}로 낮습니다. "
            "결론 문구는 '국면 라벨에 배분 정보가 검출되지 않았다'가 더 적절합니다."
        )
        tag = "diffuse"
    return verdict, top_share, tag


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--actual-path", default="tests/analysis_outputs/regime_shuffle_v2_actual_path.csv")
    parser.add_argument("--returns", default="data/assets/monthly_returns.csv")
    parser.add_argument("--labels", default="data/macro/regime_labels.csv")
    parser.add_argument("--out-dir", default="tests/analysis_outputs")
    args = parser.parse_args()

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_inputs(ROOT / args.actual_path, ROOT / args.returns, ROOT / args.labels)
    by_regime = summarize_by_regime(df)
    by_year = summarize_by_year(df)
    worst_months = (
        df.sort_values("gap_vs_6040")
        .head(10)[
            [
                "actual",
                "static_6040",
                "gap_vs_6040",
                "stocks",
                "bonds",
                "gold",
                "cash",
                "decision_regime",
                "label_regime",
                "eq_w",
                "turnover",
            ]
        ]
        .copy()
    )
    verdict, top_share, tag = concentration_verdict(by_year)

    by_regime.to_csv(out_dir / "regime_path_attribution_by_regime.csv", index=False, encoding="utf-8-sig")
    by_year.to_csv(out_dir / "regime_path_attribution_by_year.csv", index=False, encoding="utf-8-sig")
    worst_months.to_csv(out_dir / "regime_path_worst_months.csv", encoding="utf-8-sig")

    text = "\n".join(
        [
            "Regime Path Attribution Verdict",
            "================================",
            verdict,
            f"top2_loss_share={top_share:.4f}",
            f"verdict_tag={tag}",
            "",
            "Use this to separate label quality from fixed regime->asset mapping risk.",
        ]
    )
    (out_dir / "regime_path_verdict.txt").write_text(text, encoding="utf-8")

    print("=" * 72)
    print("Regime Path Attribution")
    print("=" * 72)
    print("[1] By decision regime (monthly means / compounds)")
    print(by_regime.round(4).to_string(index=False))
    print("\n[2] By year")
    pct_cols = ["actual", "static_6040", "gap_actual_minus_6040", "stocks", "bonds", "gold", "cash"]
    show_year = by_year.copy()
    show_year[pct_cols] = show_year[pct_cols] * 100.0
    print(show_year.round(2).tail(10).to_string(index=False))
    print("\n[3] Worst 10 months vs 60/40")
    show_worst = worst_months.copy()
    for col in ["actual", "static_6040", "gap_vs_6040", *ASSETS]:
        show_worst[col] = show_worst[col] * 100.0
    print(show_worst.round(2).to_string())
    print("\n[4] Verdict")
    print(verdict)
    print(f"\nSaved: {out_dir / 'regime_path_attribution_*.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
