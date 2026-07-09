"""Return heatmap payloads for strategy and asset monthly returns."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_STRATEGY_RETURNS = ROOT / "tests" / "strategy_monthly_returns.csv"
DEFAULT_ASSET_RETURNS = ROOT / "tests" / "asset_monthly_returns.csv"


def _read_returns(path: Path, *, limit_months: int | None) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    frame = frame.apply(pd.to_numeric, errors="coerce")
    frame = frame.dropna(how="all")
    if limit_months and limit_months > 0:
        frame = frame.tail(limit_months)
    return frame


def _summarize(series: pd.Series) -> dict[str, Any]:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return {"count": 0, "cumulative_pct": None, "avg_monthly_pct": None, "hit_rate_pct": None}
    cumulative = (1.0 + values).prod() - 1.0
    return {
        "count": int(len(values)),
        "cumulative_pct": round(cumulative * 100.0, 2),
        "avg_monthly_pct": round(values.mean() * 100.0, 2),
        "hit_rate_pct": round(float((values > 0).mean()) * 100.0, 1),
    }


def _max_drawdown(values: pd.Series) -> float | None:
    values = pd.to_numeric(values, errors="coerce").dropna()
    if values.empty:
        return None
    equity = (1.0 + values).cumprod()
    peak = equity.cummax()
    drawdown = equity / peak - 1.0
    return float(drawdown.min())


def _performance_metrics(values: pd.Series) -> dict[str, Any]:
    values = pd.to_numeric(values, errors="coerce").dropna()
    if values.empty:
        return {
            "months": 0,
            "cum_return_pct": None,
            "cagr_pct": None,
            "volatility_pct": None,
            "mdd_pct": None,
            "sharpe": None,
            "hit_rate_pct": None,
        }

    months = len(values)
    cumulative = (1.0 + values).prod() - 1.0
    years = months / 12.0
    cagr = (1.0 + cumulative) ** (1.0 / years) - 1.0 if years > 0 and cumulative > -1 else np.nan
    volatility = values.std(ddof=0) * np.sqrt(12.0)
    sharpe = (values.mean() * 12.0 / volatility) if volatility and volatility > 0 else np.nan
    mdd = _max_drawdown(values)
    return {
        "months": int(months),
        "cum_return_pct": round(float(cumulative) * 100.0, 2),
        "cagr_pct": None if pd.isna(cagr) else round(float(cagr) * 100.0, 2),
        "volatility_pct": round(float(volatility) * 100.0, 2),
        "mdd_pct": None if mdd is None else round(float(mdd) * 100.0, 2),
        "sharpe": None if pd.isna(sharpe) else round(float(sharpe), 3),
        "hit_rate_pct": round(float((values > 0).mean()) * 100.0, 1),
    }


def _rolling_snapshot(frame: pd.DataFrame, windows: tuple[int, ...] = (36, 24)) -> dict[str, list[dict[str, Any]]]:
    snapshots: dict[str, list[dict[str, Any]]] = {}
    for window in windows:
        rows: list[dict[str, Any]] = []
        for col in frame.columns:
            values = pd.to_numeric(frame[col], errors="coerce").dropna().tail(window)
            metrics = _performance_metrics(values)
            row = {"name": str(col), "window_months": window}
            row.update(metrics)
            rows.append(row)
        rows.sort(key=lambda r: (r.get("sharpe") is not None, r.get("sharpe") or -999), reverse=True)
        snapshots[str(window)] = rows
    return snapshots


def _review_cards(rolling: dict[str, list[dict[str, Any]]]) -> list[dict[str, str]]:
    """Rule-based human-in-the-loop review cards.

    These cards are intentionally simple and threshold-based. Their role is to
    guide the final human review, not to replace the statistical tests.
    """

    rows = [row for row in rolling.get("36", []) if row.get("months", 0) > 0]
    cards: list[dict[str, str]] = []
    if not rows:
        return cards

    sharpe_rows = [row for row in rows if row.get("sharpe") is not None]
    if sharpe_rows:
        top = max(sharpe_rows, key=lambda row: row.get("sharpe") or -999)
        if (top.get("sharpe") or 0) >= 1.0:
            cards.append(
                {
                    "level": "good",
                    "title": "위험조정 성과 우수",
                    "message": f"{top['name']}의 3년 Sharpe가 {top['sharpe']}로 1.0 이상입니다.",
                    "basis": "임계값: 3Y Sharpe >= 1.0",
                }
            )
        elif (top.get("sharpe") or 0) < 0:
            cards.append(
                {
                    "level": "warn",
                    "title": "위험조정 성과 부진",
                    "message": f"3년 기준 최고 Sharpe도 {top['sharpe']}로 0 미만입니다.",
                    "basis": "임계값: best 3Y Sharpe < 0",
                }
            )

    drawdown_rows = [row for row in rows if row.get("mdd_pct") is not None]
    if drawdown_rows:
        worst = min(drawdown_rows, key=lambda row: row.get("mdd_pct") or 0)
        if (worst.get("mdd_pct") or 0) <= -20.0:
            cards.append(
                {
                    "level": "risk",
                    "title": "MDD 확대 점검",
                    "message": f"{worst['name']}의 3년 MDD가 {worst['mdd_pct']}%입니다.",
                    "basis": "임계값: 3Y MDD <= -20%",
                }
            )

    vol_rows = [row for row in rows if row.get("volatility_pct") is not None]
    if vol_rows:
        high = max(vol_rows, key=lambda row: row.get("volatility_pct") or -999)
        if (high.get("volatility_pct") or 0) >= 30.0:
            cards.append(
                {
                    "level": "warn",
                    "title": "변동성 과다",
                    "message": f"{high['name']}의 3년 변동성이 {high['volatility_pct']}%입니다.",
                    "basis": "임계값: 3Y Volatility >= 30%",
                }
            )

    cards.append(
        {
            "level": "info",
            "title": "HILP 검토 기준",
            "message": "Rolling 지표는 경향 확인용이며, 통계적 유의성은 permutation/DSR 등 별도 관문으로 판단합니다.",
            "basis": "역할 분리: 시각화 != 통계검정",
        }
    )
    return cards


def _to_payload(frame: pd.DataFrame, *, key: str, label: str) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for dt, row in frame.iterrows():
        values: dict[str, float | None] = {}
        for col in frame.columns:
            value = row.get(col)
            values[str(col)] = None if pd.isna(value) else round(float(value) * 100.0, 2)
        rows.append({"date": pd.Timestamp(dt).strftime("%Y-%m"), "values": values})

    rolling = _rolling_snapshot(frame)
    return {
        "key": key,
        "label": label,
        "columns": [str(col) for col in frame.columns],
        "rows": rows,
        "summary": {str(col): _summarize(frame[col]) for col in frame.columns},
        "rolling": rolling,
        "review_cards": _review_cards(rolling),
    }


def build_return_heatmap(
    *,
    limit_months: int | None = 36,
    strategy_path: str | Path = DEFAULT_STRATEGY_RETURNS,
    asset_path: str | Path = DEFAULT_ASSET_RETURNS,
) -> dict[str, Any]:
    """Build strategy/asset monthly return heatmap payloads."""

    strategy = _read_returns(Path(strategy_path), limit_months=limit_months)
    asset = _read_returns(Path(asset_path), limit_months=limit_months)
    return {
        "limit_months": limit_months,
        "sets": {
            "strategy": _to_payload(strategy, key="strategy", label="전략 월수익률"),
            "asset": _to_payload(asset, key="asset", label="자산 월수익률"),
        },
        "source": {
            "strategy": str(Path(strategy_path)),
            "asset": str(Path(asset_path)),
        },
    }
