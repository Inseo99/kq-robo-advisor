"""Utilities for reading and ranking validation experiment summaries."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ValidationRow:
    mode: str
    threshold: float | None
    events: int
    edge_pct: float | None
    random_pct: float | None
    delta_vs_raw_pct: float | None
    p_value: float | None

    @property
    def edge_minus_random_pct(self) -> float | None:
        if self.edge_pct is None or self.random_pct is None:
            return None
        return self.edge_pct - self.random_pct


def load_signal_quality_summary(path: str | Path) -> list[ValidationRow]:
    """Load signal-quality validation summary rows from CSV or NPZ."""

    path = Path(path)
    if path.suffix.lower() == ".csv":
        return _rows_from_frame(pd.read_csv(path))
    if path.suffix.lower() == ".npz":
        return _rows_from_npz(path)
    raise ValueError(f"unsupported validation summary format: {path.suffix}")


def rank_quality_rows(rows: list[ValidationRow], min_events: int = 0) -> list[ValidationRow]:
    """Rank quality rows by edge-minus-random, then raw delta, then p-value."""

    candidates = [row for row in rows if row.mode == "quality" and row.events >= min_events]
    return sorted(candidates, key=_quality_key, reverse=True)


def best_quality_row(rows: list[ValidationRow], min_events: int = 0) -> ValidationRow | None:
    ranked = rank_quality_rows(rows, min_events=min_events)
    return ranked[0] if ranked else None


def format_signal_quality_markdown(
    rows: list[ValidationRow],
    *,
    title: str,
    min_events: int = 0,
) -> str:
    """Format signal-quality summary rows as a compact Markdown report."""

    best = best_quality_row(rows, min_events=min_events)
    lines = [
        f"# {title}",
        "",
        "| mode | threshold | events | edge % | random % | delta vs raw % | p-value | edge-random % |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row.mode,
                    "" if row.threshold is None else f"{row.threshold:.2f}",
                    str(row.events),
                    _fmt(row.edge_pct),
                    _fmt(row.random_pct),
                    _fmt(row.delta_vs_raw_pct),
                    _fmt(row.p_value, digits=4),
                    _fmt(row.edge_minus_random_pct),
                ]
            )
            + " |"
        )
    if best is not None:
        lines.extend(
            [
                "",
                (
                    f"Best quality candidate with at least {min_events} events: "
                    f"q>={best.threshold:.2f}, edge {_fmt(best.edge_pct)}%, "
                    f"random {_fmt(best.random_pct)}%, events {best.events}."
                ),
            ]
        )
    return "\n".join(lines) + "\n"


def _quality_key(row: ValidationRow) -> tuple[float, float, float]:
    edge_minus_random = row.edge_minus_random_pct
    delta = row.delta_vs_raw_pct
    p_value = row.p_value
    return (
        -1e9 if edge_minus_random is None else edge_minus_random,
        -1e9 if delta is None else delta,
        -1e9 if p_value is None else -p_value,
    )


def _fmt(value: float | None, digits: int = 3) -> str:
    return "" if value is None else f"{value:.{digits}f}"


def _rows_from_frame(frame: pd.DataFrame) -> list[ValidationRow]:
    rows = []
    for _, row in frame.iterrows():
        rows.append(
            ValidationRow(
                mode=str(row["mode"]),
                threshold=_optional_float(row.get("threshold")),
                events=int(row["events"]),
                edge_pct=_optional_float(row.get("edge_pct")),
                random_pct=_optional_float(row.get("random_pct")),
                delta_vs_raw_pct=_optional_float(row.get("delta_vs_raw_pct")),
                p_value=_optional_float(row.get("p_value")),
            )
        )
    return rows


def _rows_from_npz(path: Path) -> list[ValidationRow]:
    data = np.load(path, allow_pickle=True)
    raw = data["raw_summary"].item()
    rows = [
        ValidationRow(
            mode="raw",
            threshold=None,
            events=int(raw["n_events"]),
            edge_pct=_pct(raw.get("actual_mean")),
            random_pct=_pct(raw.get("random_mean")),
            delta_vs_raw_pct=0.0,
            p_value=_optional_float(raw.get("p_value")),
        )
    ]
    raw_edge = raw.get("actual_mean")
    for item in data["quality_summaries"]:
        row = item.item() if hasattr(item, "item") else item
        summary = row["summary"]
        edge = summary.get("actual_mean")
        rows.append(
            ValidationRow(
                mode="quality",
                threshold=_optional_float(row.get("threshold")),
                events=int(summary["n_events"]),
                edge_pct=_pct(edge),
                random_pct=_pct(summary.get("random_mean")),
                delta_vs_raw_pct=None if raw_edge is None or edge is None else (edge - raw_edge) * 100,
                p_value=_optional_float(summary.get("p_value")),
            )
        )
    return rows


def _pct(value: object) -> float | None:
    value = _optional_float(value)
    return None if value is None else value * 100


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except Exception:
        return None
    if not np.isfinite(parsed):
        return None
    return parsed
