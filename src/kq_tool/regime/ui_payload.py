"""Build a single-source regime payload for the UI.

The legacy UI used to mix LightGBM confidence, HMM persistence, and hard-coded
validity rules. This module derives the customer-facing regime numbers from one
source of truth: the confirmed PiT regime labels and their smoothed transition
matrix. That keeps current regime, stay probability, validity period, early
re-evaluation triggers, and transition matrix displays internally consistent.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from .regime_labels import (
    REGIMES,
    next_quarter_transition_probs,
    regime_diagnostics,
    transition_matrix,
)

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_LABELS = ROOT / "data" / "macro" / "regime_labels.csv"
DEFAULT_OUT_DIR = ROOT / "tests" / "analysis_outputs"
DEFAULT_OUT = DEFAULT_OUT_DIR / "regime_ui_payload.json"
NOWCAST_PROBS = DEFAULT_OUT_DIR / "regime_model_probs_nowcast.csv"
VERDICT_FILE = DEFAULT_OUT_DIR / "regime_model_verdict.txt"

TRADING_DAYS_PER_MONTH = 21.0
CALENDAR_DAYS_PER_MONTH = 30.44
NAIVE_CURRENT_PROB = 0.90
NOWCAST_CONFIDENCE_FLOOR = 0.60
NEXT_QUARTER_STAY_FLOOR = 0.50

DEFAULT_REGIME_DESC = {
    "골디락스": {
        "label": "성장↑·물가↓",
        "color": "#3fb950",
        "desc": "주식 강세, 채권 중립",
        "best_assets": ["주식", "성장주", "기술주"],
    },
    "리플레이션": {
        "label": "성장↑·물가↑",
        "color": "#d29922",
        "desc": "원자재 수혜, 채권 약세",
        "best_assets": ["원자재", "에너지", "금융주", "가치주"],
    },
    "스태그플레이션": {
        "label": "성장↓·물가↑",
        "color": "#f85149",
        "desc": "금·원자재 헤지, 현금 방어",
        "best_assets": ["금", "원자재", "현금", "필수소비재"],
    },
    "디플레이션": {
        "label": "성장↓·물가↓",
        "color": "#388bfd",
        "desc": "국채 강세, 방어주 선호",
        "best_assets": ["국채", "방어주", "유틸리티", "현금"],
    },
}


def _month_index(obj: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    out = obj.copy()
    out.index = pd.to_datetime(out.index).to_period("M").to_timestamp("M")
    return out.sort_index()


def load_labels(path: Path = DEFAULT_LABELS) -> pd.Series:
    """Load confirmed monthly labels from `date,regime` CSV."""

    if not path.exists():
        raise FileNotFoundError(path)

    frame = pd.read_csv(path, encoding="utf-8-sig")
    if "date" in frame.columns:
        frame["date"] = pd.to_datetime(frame["date"])
        frame = frame.set_index("date")
    else:
        frame = pd.read_csv(path, index_col=0, parse_dates=True, encoding="utf-8-sig")

    regime_col = "regime" if "regime" in frame.columns else frame.columns[0]
    labels = _month_index(frame[regime_col].astype("string")).dropna()
    labels = labels[labels.isin(REGIMES)]
    if labels.empty:
        raise ValueError(f"No valid regime labels found in {path}")
    labels.name = "regime"
    return labels


def _load_latest_probs(path: Path, expected_date: pd.Timestamp) -> dict[str, float] | None:
    if not path.exists():
        return None

    frame = pd.read_csv(path, index_col=0, parse_dates=True, encoding="utf-8-sig")
    if frame.empty:
        return None

    frame = _month_index(frame)
    latest_date = frame.index[-1]
    if latest_date != expected_date:
        return None

    row = frame.iloc[-1].reindex(REGIMES).astype(float).fillna(0.0)
    total = float(row.sum())
    if total <= 0:
        return None
    return {regime: float(row[regime] / total) for regime in REGIMES}


def _naive_probs(current_regime: str) -> dict[str, float]:
    other_prob = (1.0 - NAIVE_CURRENT_PROB) / (len(REGIMES) - 1)
    return {
        regime: (NAIVE_CURRENT_PROB if regime == current_regime else other_prob)
        for regime in REGIMES
    }


def _prob_dict(series: pd.Series) -> dict[str, float]:
    row = series.reindex(REGIMES).astype(float).fillna(0.0)
    total = float(row.sum())
    if total > 0:
        row = row / total
    return {regime: float(row[regime]) for regime in REGIMES}


def _transition_payload(labels: pd.Series) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    probs, counts = transition_matrix(labels, alpha=1.0)
    payload: dict[str, Any] = {}
    for before in REGIMES:
        payload[before] = {}
        for after in REGIMES:
            prob = float(probs.loc[before, after])
            n = int(counts.loc[before, after])
            payload[before][after] = {
                "prob": prob,
                "n": n,
                "display": f"{prob * 100:.1f}% (n={n})",
            }
    return payload, probs, counts


def _current_run_months(labels: pd.Series) -> int:
    clean = labels.dropna()
    if clean.empty:
        return 0
    current = clean.iloc[-1]
    months = 0
    for value in reversed(clean.tolist()):
        if value != current:
            break
        months += 1
    return months


def _expected_remaining(monthly_stay_prob: float) -> dict[str, float | int | str]:
    p = min(max(float(monthly_stay_prob), 0.0), 0.999)
    months = p / max(1.0 - p, 1e-9)
    calendar_days = int(round(months * CALENDAR_DAYS_PER_MONTH))
    trading_days = int(round(months * TRADING_DAYS_PER_MONTH))
    return {
        "months": round(months, 2),
        "calendar_days": calendar_days,
        "trading_days": trading_days,
        "display": f"약 {calendar_days}일",
    }


def _read_method_label(path: Path = VERDICT_FILE) -> tuple[str, dict[str, str | None]]:
    selected = {"nowcast": None, "forecast": None}
    if path.exists():
        text = path.read_text(encoding="utf-8")
        now = re.search(r"nowcast:.*selected=([A-Za-z0-9_^]+)", text)
        fore = re.search(r"forecast_\d+m:.*selected=([A-Za-z0-9_^]+)", text)
        if now:
            selected["nowcast"] = now.group(1)
        if fore:
            selected["forecast"] = fore.group(1)

    if selected["nowcast"] == "naive" and selected["forecast"] == "p3":
        label = "판정: 지속성 베이스라인 + P^3 전이행렬 (M4 검증 완료)"
    elif selected["nowcast"] == "model" or selected["forecast"] == "model":
        label = "판정: LightGBM 확률 채택 (walk-forward 검증 통과)"
    else:
        label = "판정: 지속성 베이스라인 (검증 완료)"
    return label, selected


def build_payload(
    labels_path: Path = DEFAULT_LABELS,
    nowcast_probs_path: Path = NOWCAST_PROBS,
    verdict_path: Path = VERDICT_FILE,
) -> dict[str, Any]:
    """Build the JSON-serializable UI payload."""

    labels = load_labels(labels_path)
    current_date = labels.index[-1]
    current_regime = str(labels.iloc[-1])

    transition, monthly_probs, counts = _transition_payload(labels)
    p3 = next_quarter_transition_probs(labels, alpha=1.0)
    next_quarter_probs = _prob_dict(p3.loc[current_regime])

    nowcast_probs = _load_latest_probs(nowcast_probs_path, current_date)
    nowcast_source = "selected_nowcast_csv"
    if nowcast_probs is None:
        nowcast_probs = _naive_probs(current_regime)
        nowcast_source = "naive_from_latest_label"

    stay_prob_monthly = float(monthly_probs.loc[current_regime, current_regime])
    stay_prob_next_quarter = float(next_quarter_probs[current_regime])
    expected_remaining = _expected_remaining(stay_prob_monthly)

    nowcast_confidence = max(nowcast_probs.values())
    reeval_reasons: list[str] = []
    if nowcast_confidence < NOWCAST_CONFIDENCE_FLOOR:
        reeval_reasons.append(
            f"현재 국면 확률 {nowcast_confidence * 100:.1f}% < {NOWCAST_CONFIDENCE_FLOOR * 100:.0f}%"
        )
    if stay_prob_next_quarter < NEXT_QUARTER_STAY_FLOOR:
        reeval_reasons.append(
            f"다음 분기 유지확률 {stay_prob_next_quarter * 100:.1f}% < {NEXT_QUARTER_STAY_FLOOR * 100:.0f}%"
        )
    if not reeval_reasons:
        reeval_reasons.append("전이확률 기준 조기 재평가 트리거 없음")

    method_label, selected_sources = _read_method_label(verdict_path)
    diagnostics = regime_diagnostics(labels)

    payload: dict[str, Any] = {
        "asof": current_date.strftime("%Y-%m-%d"),
        "current_regime": current_regime,
        "current_run_months": _current_run_months(labels),
        "method_label": method_label,
        "selected_sources": selected_sources,
        "nowcast_source": nowcast_source,
        "nowcast_probs": nowcast_probs,
        "next_quarter_probs": next_quarter_probs,
        "stay_prob_monthly": stay_prob_monthly,
        "stay_prob_next_quarter": stay_prob_next_quarter,
        "expected_remaining_months": expected_remaining["months"],
        "expected_remaining_days": expected_remaining["calendar_days"],
        "expected_remaining_trading_days": expected_remaining["trading_days"],
        "validity_display": expected_remaining["display"],
        "reeval_flag": any(reason != "전이확률 기준 조기 재평가 트리거 없음" for reason in reeval_reasons),
        "reeval_reasons": reeval_reasons,
        "transition_matrix": transition,
        "diagnostics": diagnostics,
        "counts_total": int(counts.to_numpy().sum()),
        "source_files": {
            "labels": str(labels_path),
            "nowcast_probs": str(nowcast_probs_path),
            "verdict": str(verdict_path),
        },
    }
    return payload


def write_payload(payload: dict[str, Any], out_path: Path = DEFAULT_OUT) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", default=str(DEFAULT_LABELS))
    parser.add_argument("--nowcast-probs", default=str(NOWCAST_PROBS))
    parser.add_argument("--verdict", default=str(VERDICT_FILE))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args(argv)

    payload = build_payload(
        labels_path=Path(args.labels),
        nowcast_probs_path=Path(args.nowcast_probs),
        verdict_path=Path(args.verdict),
    )
    out = write_payload(payload, Path(args.out))
    print(f"saved: {out}")
    print(
        f"current={payload['current_regime']} "
        f"nowcast={max(payload['nowcast_probs'].values()) * 100:.1f}% "
        f"stay_q={payload['stay_prob_next_quarter'] * 100:.1f}% "
        f"validity={payload['validity_display']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

