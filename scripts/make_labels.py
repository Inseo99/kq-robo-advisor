r"""Create and export PiT-safe regime labels.

Pipeline:
    data/macro/*.csv
      -> macro_data.get_observable_panel()
      -> regime_labels.make_regime_labels()
      -> diagnostics + data/macro/regime_labels.csv
      -> optional KOSPI regime-colored chart

Usage:
    python scripts\make_labels.py
    python scripts\make_labels.py --confirm 3 --momentum 6
    python scripts\make_labels.py --chart-only

Sanity guidance for real data:
    Average duration around 3-4 quarters and roughly 10-15 transitions over
    20 years is a useful starting point. If parameters are changed, record the
    number of attempts for multiple-testing correction. Do not tune from
    portfolio performance.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kq_tool.regime.macro_data import get_observable_panel  # noqa: E402
from kq_tool.regime.regime_labels import (  # noqa: E402
    REGIMES,
    LabelConfig,
    make_regime_labels,
    next_quarter_transition_probs,
    regime_diagnostics,
    transition_matrix,
)

OUT_CSV = ROOT / "data" / "macro" / "regime_labels.csv"
OUT_CHART = ROOT / "tests" / "analysis_outputs" / "regime_kospi_chart.png"

REGIME_COLORS = {
    "골디락스": "#2e7d32",
    "리플레이션": "#b8860b",
    "스태그플레이션": "#b71c1c",
    "디플레이션": "#1565c0",
}


def make_chart(labels: pd.Series) -> bool:
    """Save a KOSPI chart with regime-colored background bands."""

    try:
        kospi = get_observable_panel(["kospi"])["kospi"].dropna()
    except FileNotFoundError:
        print("[skip] data/macro/kospi.csv not found; chart skipped")
        return False

    common = labels.dropna().index.intersection(kospi.index)
    if len(common) < 2:
        print("[skip] not enough common KOSPI/regime dates for chart")
        return False

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.patches import Patch

    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    for candidate in ["Malgun Gothic", "AppleGothic", "NanumGothic", "NanumBarunGothic"]:
        if candidate in available_fonts:
            plt.rcParams["font.family"] = candidate
            break
    plt.rcParams["axes.unicode_minus"] = False

    regime = labels.loc[common]
    price = kospi.loc[common]

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(price.index, price.to_numpy(dtype=float), color="black", linewidth=1.2)
    ax.set_yscale("log")

    run_start = common[0]
    for i in range(1, len(common) + 1):
        end_run = i == len(common) or regime.iloc[i] != regime.iloc[i - 1]
        if not end_run:
            continue
        name = str(regime.iloc[i - 1])
        run_end = common[min(i, len(common) - 1)]
        ax.axvspan(run_start, run_end, color=REGIME_COLORS.get(name, "#888888"), alpha=0.18)
        if i < len(common):
            run_start = common[i]

    ax.legend(
        handles=[Patch(color=color, alpha=0.4, label=name) for name, color in REGIME_COLORS.items()],
        loc="upper left",
        fontsize=9,
    )
    ax.set_title("KOSPI x PiT Regime Labels")
    OUT_CHART.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT_CHART, dpi=130)
    plt.close(fig)
    print(f"chart saved: {OUT_CHART}")
    return True


def print_diagnostics(labels: pd.Series, cfg: LabelConfig) -> None:
    clean = labels.dropna()
    if clean.empty:
        raise RuntimeError("No regime labels were produced. Check macro input coverage.")

    diagnostics = regime_diagnostics(labels)
    print("=" * 64)
    print(
        f"Regime label diagnostics "
        f"(confirm={cfg.confirm_months}, momentum={cfg.momentum_window}, warmup={cfg.warmup_months})"
    )
    print("=" * 64)
    print(f"  period              : {clean.index[0]:%Y-%m} ~ {clean.index[-1]:%Y-%m} ({diagnostics['n_months']} months)")
    print(f"  transitions         : {diagnostics['n_transitions']}")
    print(f"  avg duration qtrs   : {diagnostics['avg_duration_quarters']:.2f}")
    by_regime = diagnostics["avg_duration_by_regime_q"]
    print("  duration by regime  : " + ", ".join(f"{name} {duration:.1f}" for name, duration in by_regime.items()))
    print(f"  regime share        : {diagnostics['regime_share']}")

    avg_duration = float(diagnostics["avg_duration_quarters"])
    years = diagnostics["n_months"] / 12.0
    transitions_per_year = diagnostics["n_transitions"] / years if years > 0 else float("nan")
    ok_duration = 3.0 <= avg_duration <= 6.0
    ok_transitions = 0.4 <= transitions_per_year <= 1.0
    print(
        "  sanity             : "
        f"duration {'OK' if ok_duration else 'OUT-OF-RANGE'}, "
        f"transition frequency {'OK' if ok_transitions else 'OUT-OF-RANGE'}"
    )

    print("\nTransition matrix, smoothed probability with raw n:")
    probs, counts = transition_matrix(labels)
    for before in REGIMES:
        cells = [f"{probs.loc[before, after] * 100:4.0f}%(n={counts.loc[before, after]})" for after in REGIMES]
        print(f"  {before:<8} -> " + "  ".join(cells))

    current = clean.iloc[-1]
    next_probs = next_quarter_transition_probs(labels).loc[current]
    print(
        f"\nCurrent regime: {current} / next-quarter P^3: "
        + ", ".join(f"{name} {prob * 100:.1f}%" for name, prob in next_probs.items())
    )


def export_labels(labels: pd.Series) -> None:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    labels.dropna().rename("regime").rename_axis("date").to_csv(OUT_CSV, encoding="utf-8-sig")
    print(f"\nlabels saved: {OUT_CSV}")
    print(
        "next: python tests\\validation_regime_shuffle_v2.py "
        "--labels data/macro/regime_labels.csv --returns data/assets/monthly_returns.csv"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--confirm", type=int, default=3)
    parser.add_argument("--momentum", type=int, default=6)
    parser.add_argument("--warmup", type=int, default=24)
    parser.add_argument("--chart-only", action="store_true")
    args = parser.parse_args()

    cfg = LabelConfig(
        confirm_months=args.confirm,
        momentum_window=args.momentum,
        warmup_months=args.warmup,
    )

    try:
        labels = make_regime_labels(cfg)
    except FileNotFoundError as exc:
        print(f"[stop] macro data missing: {exc}")
        print("       Run scripts/fetch_ecos.py first, then tests/validation_macro_pit.py.")
        return 1

    try:
        print_diagnostics(labels, cfg)
    except RuntimeError as exc:
        print(f"[stop] {exc}")
        return 1

    if not args.chart_only:
        export_labels(labels)
    make_chart(labels)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

