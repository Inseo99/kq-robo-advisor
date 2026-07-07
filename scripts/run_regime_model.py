r"""Run real-data regime model verdicts and export UI-ready probabilities.

Inputs:
  - data/macro/regime_labels.csv
  - data/macro/*.csv market feature files loaded through macro_data

Outputs in tests/analysis_outputs:
  - regime_model_probs_nowcast.csv          selected nowcast source
  - regime_model_probs_nowcast_model.csv    raw model nowcast
  - regime_model_probs_nowcast_naive.csv    persistence baseline
  - regime_model_probs_forecast.csv         selected 3-month forecast source
  - regime_model_probs_forecast_model.csv   raw model forecast
  - regime_model_probs_forecast_p3.csv      transition baseline forecast
  - regime_transition_matrix.csv            long form with display text
  - regime_model_verdict.txt                concise adoption verdict
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kq_tool.regime.macro_data import get_observable_panel  # noqa: E402
from kq_tool.regime.regime_labels import REGIMES, transition_matrix  # noqa: E402
from kq_tool.regime.regime_model_v2 import (  # noqa: E402
    ModelConfig,
    build_features,
    calibration_table,
    log_loss_score,
    naive_nowcast_baseline,
    p3_forecast_baseline,
    walk_forward_predict,
)

OUT_DIR = ROOT / "tests" / "analysis_outputs"
FEATURE_KEYS = ["yield_spread_10y_3y", "usdkrw", "credit_spread", "kospi", "base_rate"]


def _month_index(obj: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    out = obj.copy()
    out.index = pd.to_datetime(out.index).to_period("M").to_timestamp("M")
    return out.sort_index()


def load_labels(path: Path) -> pd.Series:
    frame = pd.read_csv(path, parse_dates=["date"]).set_index("date")
    regime_col = "regime" if "regime" in frame.columns else frame.columns[0]
    labels = frame[regime_col].astype("string")
    labels = _month_index(labels).dropna()
    return labels[labels.isin(REGIMES)]


def save_probs(frame: pd.DataFrame, path: Path) -> None:
    out = frame.copy().astype(float)
    out.index.name = "date"
    out.to_csv(path, encoding="utf-8-sig")


def save_transition(labels: pd.Series, path: Path) -> None:
    probs, counts = transition_matrix(labels, alpha=1.0)
    rows = []
    for before in REGIMES:
        for after in REGIMES:
            prob = float(probs.loc[before, after])
            n = int(counts.loc[before, after])
            rows.append(
                {
                    "from": before,
                    "to": after,
                    "prob": prob,
                    "n": n,
                    "display": f"{prob * 100:.1f}% (n={n})",
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


def evaluate_nowcast(features: pd.DataFrame, labels: pd.Series, min_train_months: int) -> dict:
    cfg = ModelConfig(task="nowcast", min_train_months=min_train_months)
    model_probs, audit = walk_forward_predict(features, labels, cfg)
    naive = naive_nowcast_baseline(labels, cfg)
    common = model_probs.index.intersection(naive.index).intersection(labels.index)
    model_probs = model_probs.loc[common]
    naive = naive.loc[common]
    ll_model = log_loss_score(model_probs, labels)
    ll_naive = log_loss_score(naive, labels)
    selected_name = "model" if np.isfinite(ll_model) and ll_model < ll_naive else "naive"
    selected = model_probs if selected_name == "model" else naive
    return {
        "cfg": cfg,
        "model": model_probs,
        "baseline": naive,
        "selected": selected,
        "audit": audit,
        "ll_model": ll_model,
        "ll_baseline": ll_naive,
        "selected_name": selected_name,
        "calibration": calibration_table(model_probs, labels),
    }


def evaluate_forecast(features: pd.DataFrame, labels: pd.Series, min_train_months: int, horizon_m: int) -> dict:
    cfg = ModelConfig(task="forecast", horizon_m=horizon_m, min_train_months=min_train_months)
    model_probs, audit = walk_forward_predict(features, labels, cfg)
    p3 = p3_forecast_baseline(labels, cfg)
    actual = labels.shift(-horizon_m)
    common = model_probs.index.intersection(p3.index).intersection(actual.dropna().index)
    model_probs = model_probs.loc[common]
    p3 = p3.loc[common]
    ll_model = log_loss_score(model_probs, actual)
    ll_p3 = log_loss_score(p3, actual)
    selected_name = "model" if np.isfinite(ll_model) and ll_model < ll_p3 else "p3"
    selected = model_probs if selected_name == "model" else p3
    return {
        "cfg": cfg,
        "model": model_probs,
        "baseline": p3,
        "selected": selected,
        "audit": audit,
        "ll_model": ll_model,
        "ll_baseline": ll_p3,
        "selected_name": selected_name,
        "calibration": calibration_table(model_probs, actual),
    }


def latest_line(name: str, probs: pd.DataFrame) -> str:
    if probs.empty:
        return f"{name}: no probabilities"
    row = probs.iloc[-1].sort_values(ascending=False)
    return f"{name} latest {probs.index[-1]:%Y-%m}: " + ", ".join(
        f"{regime} {prob * 100:.1f}%" for regime, prob in row.items()
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", default="data/macro/regime_labels.csv")
    parser.add_argument("--out-dir", default="tests/analysis_outputs")
    parser.add_argument("--min-train-months", type=int, default=60)
    parser.add_argument("--horizon", type=int, default=3)
    args = parser.parse_args()

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = load_labels(ROOT / args.labels)
    panel = get_observable_panel(FEATURE_KEYS)
    features = build_features(panel)
    features = _month_index(features)
    common = labels.index.intersection(features.index)
    labels = labels.loc[common]
    features = features.loc[common]

    now = evaluate_nowcast(features, labels, args.min_train_months)
    fore = evaluate_forecast(features, labels, args.min_train_months, args.horizon)

    save_probs(now["selected"], out_dir / "regime_model_probs_nowcast.csv")
    save_probs(now["model"], out_dir / "regime_model_probs_nowcast_model.csv")
    save_probs(now["baseline"], out_dir / "regime_model_probs_nowcast_naive.csv")
    save_probs(fore["selected"], out_dir / "regime_model_probs_forecast.csv")
    save_probs(fore["model"], out_dir / "regime_model_probs_forecast_model.csv")
    save_probs(fore["baseline"], out_dir / "regime_model_probs_forecast_p3.csv")
    save_transition(labels, out_dir / "regime_transition_matrix.csv")
    now["calibration"].to_csv(out_dir / "regime_model_calibration_nowcast.csv", encoding="utf-8-sig")
    fore["calibration"].to_csv(out_dir / "regime_model_calibration_forecast.csv", encoding="utf-8-sig")

    verdict_lines = [
        "Regime Model Real-Data Verdict",
        "==============================",
        f"labels: {labels.index[0]:%Y-%m} ~ {labels.index[-1]:%Y-%m}, n={len(labels)}",
        f"nowcast: model logloss={now['ll_model']:.4f}, naive={now['ll_baseline']:.4f}, selected={now['selected_name']}",
        f"forecast_{args.horizon}m: model logloss={fore['ll_model']:.4f}, P^3={fore['ll_baseline']:.4f}, selected={fore['selected_name']}",
        latest_line("nowcast_selected", now["selected"]),
        latest_line("forecast_selected", fore["selected"]),
        "",
        "Adoption rule: use model probabilities only when walk-forward log-loss beats the pre-registered baseline.",
    ]
    (out_dir / "regime_model_verdict.txt").write_text("\n".join(verdict_lines), encoding="utf-8")

    print("=" * 72)
    print("Regime Model Real-Data Verdict")
    print("=" * 72)
    print(verdict_lines[2])
    print(verdict_lines[3])
    print(verdict_lines[4])
    print(verdict_lines[5])
    print(verdict_lines[6])
    print("\nSaved UI outputs:")
    for filename in [
        "regime_model_probs_nowcast.csv",
        "regime_model_probs_forecast.csv",
        "regime_transition_matrix.csv",
        "regime_model_verdict.txt",
    ]:
        print(f"  {out_dir / filename}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
