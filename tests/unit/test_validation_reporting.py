from __future__ import annotations

import numpy as np
import pandas as pd

from kq_tool.validation.reporting import (
    best_quality_row,
    format_signal_quality_markdown,
    load_signal_quality_summary,
    rank_quality_rows,
)


def test_load_signal_quality_summary_from_csv_and_rank(tmp_path) -> None:
    path = tmp_path / "summary.csv"
    pd.DataFrame(
        [
            {
                "mode": "raw",
                "threshold": "",
                "events": 100,
                "edge_pct": 1.0,
                "random_pct": 0.5,
                "delta_vs_raw_pct": 0.0,
                "p_value": 0.01,
            },
            {
                "mode": "quality",
                "threshold": 0.6,
                "events": 80,
                "edge_pct": 1.4,
                "random_pct": 0.4,
                "delta_vs_raw_pct": 0.4,
                "p_value": 0.02,
            },
            {
                "mode": "quality",
                "threshold": 0.7,
                "events": 20,
                "edge_pct": 1.8,
                "random_pct": 1.0,
                "delta_vs_raw_pct": 0.8,
                "p_value": 0.03,
            },
        ]
    ).to_csv(path, index=False)

    rows = load_signal_quality_summary(path)
    ranked = rank_quality_rows(rows, min_events=50)

    assert len(rows) == 3
    assert ranked[0].threshold == 0.6
    assert best_quality_row(rows, min_events=50).threshold == 0.6
    markdown = format_signal_quality_markdown(rows, title="Example", min_events=50)
    assert "# Example" in markdown
    assert "q>=0.60" in markdown


def test_load_signal_quality_summary_from_npz(tmp_path) -> None:
    path = tmp_path / "summary.npz"
    np.savez_compressed(
        path,
        raw_summary={
            "n_events": 100,
            "actual_mean": 0.01,
            "random_mean": 0.005,
            "p_value": 0.01,
        },
        quality_summaries=np.array(
            [
                {
                    "threshold": 0.7,
                    "summary": {
                        "n_events": 60,
                        "actual_mean": 0.018,
                        "random_mean": 0.004,
                        "p_value": 0.0,
                    },
                }
            ],
            dtype=object,
        ),
    )

    rows = load_signal_quality_summary(path)
    best = best_quality_row(rows)

    assert rows[0].mode == "raw"
    assert rows[0].edge_pct == 1.0
    assert best is not None
    assert best.threshold == 0.7
    assert round(best.delta_vs_raw_pct, 3) == 0.8
