"""One-click DSR + regime-conditional performance runner.

Reads the three CSV files produced by tests/export_dsr_inputs.py and writes:
  1) Deflated Sharpe Ratio table
  2) Sharpe 95% bootstrap CI table
  3) regime x strategy / regime x asset performance matrices
  4) regime-level excess return t-tests versus a benchmark, when available

Outputs are saved under data/analysis_outputs/ by default.

Example:
  python tests/run_dsr_regime_analysis.py --data-dir tests --n-trials 15 --benchmark KOSPI
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT / "src"), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from kq_tool.validation.deflated_sharpe import (  # noqa: E402
    bootstrap_sharpe_ci,
    deflated_sharpe_table,
)
from kq_tool.validation.regime_performance import run_full_decomposition  # noqa: E402


INPUT_FILES = (
    "strategy_monthly_returns.csv",
    "asset_monthly_returns.csv",
    "regime_labels_monthly.csv",
)


def _input_paths(data_dir: str | Path) -> tuple[Path, Path, Path, Path]:
    """Resolve input CSVs, falling back to tests/ for current project layout."""
    requested = Path(data_dir)
    requested = requested if requested.is_absolute() else ROOT / requested

    candidates = [requested]
    tests_dir = ROOT / "tests"
    if tests_dir not in candidates:
        candidates.append(tests_dir)
    if ROOT not in candidates:
        candidates.append(ROOT)

    for candidate in candidates:
        paths = [candidate / name for name in INPUT_FILES]
        if all(path.exists() for path in paths):
            if candidate != requested:
                print(f"[info] 입력 CSV를 {requested}에서 찾지 못해 {candidate}를 사용합니다.")
            return candidate, paths[0], paths[1], paths[2]

    checked = ", ".join(str(c) for c in candidates)
    raise FileNotFoundError(f"Required CSV inputs not found. Checked: {checked}")


def _resolve_benchmark(requested: str | None, columns: list[str]) -> str | None:
    """Map user-friendly benchmark names like KOSPI to an existing column."""
    if not requested:
        return None
    if requested in columns:
        return requested

    needle = requested.lower().strip()
    for col in columns:
        low = str(col).lower()
        if needle in low or low in needle:
            print(f"[info] benchmark '{requested}' -> '{col}'")
            return str(col)

    print(f"[warn] benchmark '{requested}' 컬럼을 찾지 못했습니다. 초과수익 t-test는 가능한 표에서만 건너뜁니다.")
    return requested


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="tests", help="CSV input folder. Defaults to tests/.")
    parser.add_argument("--out-dir", default=None, help="Output folder. Defaults to data/analysis_outputs.")
    parser.add_argument("--n-trials", type=int, default=15, help="Multiple-comparison trial count for DSR.")
    parser.add_argument("--benchmark", default=None, help="Benchmark column name, e.g. KOSPI.")
    parser.add_argument("--min-months", type=int, default=12)
    parser.add_argument("--bootstrap", type=int, default=10_000, help="Bootstrap repetitions for Sharpe CI.")
    args = parser.parse_args()

    _, strat_csv, asset_csv, regime_csv = _input_paths(args.data_dir)
    out_dir = Path(args.out_dir) if args.out_dir else ROOT / "data" / "analysis_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    strat = pd.read_csv(strat_csv, index_col=0, parse_dates=True).sort_index()
    benchmark = _resolve_benchmark(args.benchmark, list(strat.columns))

    dsr = deflated_sharpe_table(strat, n_trials=args.n_trials)
    dsr.to_csv(out_dir / "dsr_table.csv", encoding="utf-8-sig")

    print("=" * 72)
    print(f"[1] Deflated Sharpe Ratio (N_trials={args.n_trials})")
    print("=" * 72)
    show_cols = ["sr_annualized", "psr_vs_zero", "expected_max_sr_annualized", "dsr", "significant_5pct"]
    show = dsr[show_cols].copy()
    for col in show_cols:
        if col != "significant_5pct":
            show[col] = pd.to_numeric(show[col], errors="coerce")
    print(show.round(3).to_string())

    ci_rows = {
        col: bootstrap_sharpe_ci(strat[col].dropna(), n_boot=args.bootstrap)
        for col in strat.columns
    }
    ci = pd.DataFrame(ci_rows).T[["sharpe_annualized", "ci_low", "ci_high"]].round(3)
    ci.to_csv(out_dir / "sharpe_bootstrap_ci.csv", encoding="utf-8-sig")

    print()
    print("[2] Sharpe 95% Bootstrap CI")
    print(ci.to_string())

    results = run_full_decomposition(
        strat_csv,
        regime_csv,
        asset_csv=asset_csv if asset_csv.exists() else None,
        benchmark=benchmark,
        out_dir=out_dir,
        min_months=args.min_months,
    )

    print()
    print("[3] Regime x Strategy Annualized Sharpe")
    print(results["strategy_sharpe_pivot"].to_string())

    if not results["best_per_regime"].empty:
        print()
        print("[3b] Best Strategy Per Regime (Sharpe, sufficient regimes only)")
        print(results["best_per_regime"].round(3).to_string(index=False))

    for key, title in (
        ("strategy_excess_vs_benchmark", "Strategy"),
        ("asset_excess_vs_benchmark", "Asset"),
    ):
        if key in results:
            print()
            print(f"[4] {title} Excess vs {benchmark} (annualized pp, paired t)")
            top = results[key][~results[key]["insufficient"]].round(3)
            print(top.to_string(index=False))

    print()
    print(f"모든 결과 저장 완료 -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
