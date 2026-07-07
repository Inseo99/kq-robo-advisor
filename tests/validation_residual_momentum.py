"""Residual Momentum validation gate.

Run:
    python tests\validation_residual_momentum.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kq_tool.data.gateway import load_close_panel  # noqa: E402
from kq_tool.portfolio.residual_momentum import (  # noqa: E402
    FACTOR_COLS,
    ResMomConfig,
    compute_residual_momentum_signal,
    run_backtest,
)

OUT = ROOT / "tests" / "analysis_outputs" / "residual_momentum_result.json"
FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  -- {detail}" if detail and not cond else ""))
    if not cond:
        FAILURES.append(name)


def demo_inputs(n_stocks: int = 60, n_months: int = 120, seed: int = 7) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2015-01-31", periods=n_months, freq="ME")
    mkt = rng.normal(0.006, 0.05, n_months)
    smb = rng.normal(0.001, 0.02, n_months)
    hml = rng.normal(0.001, 0.02, n_months)
    factors = pd.DataFrame({"MKT": mkt, "SMB": smb, "HML": hml, "RF": 0.002}, index=dates)

    beta_m = rng.uniform(0.5, 1.5, n_stocks)
    beta_s = rng.uniform(-0.5, 0.8, n_stocks)
    beta_h = rng.uniform(-0.5, 0.8, n_stocks)
    eps = np.zeros((n_months, n_stocks))
    for t in range(1, n_months):
        eps[t] = 0.35 * eps[t - 1] + rng.normal(0, 0.04, n_stocks)
    returns = (
        0.002
        + np.outer(mkt, beta_m)
        + np.outer(smb, beta_s)
        + np.outer(hml, beta_h)
        + eps
    )
    frame = pd.DataFrame(returns, index=dates, columns=[f"S{i:03d}" for i in range(n_stocks)])
    frame.iloc[-18:, 0] = np.nan
    return frame, factors


def load_real_inputs(max_cols: int = 120) -> tuple[pd.DataFrame, pd.DataFrame]:
    close = load_close_panel()
    monthly = close.resample("ME").last().pct_change().replace([np.inf, -np.inf], np.nan)
    cols = monthly.notna().sum().sort_values(ascending=False).head(max_cols).index
    monthly = monthly[cols].dropna(how="all")
    factors = pd.read_csv(ROOT / "tests" / "factor_returns_korea.csv", index_col=0, parse_dates=True)
    return monthly, factors


def main() -> int:
    print("=" * 72)
    print("Residual Momentum validation")
    print("=" * 72)
    returns, factors = demo_inputs()
    cfg = ResMomConfig(est_window=36, top_n=10, min_obs=30)
    asof = returns.index[-1]
    signal = compute_residual_momentum_signal(returns, factors, asof, cfg)

    asof_mid = returns.index[-13]
    signal_mid = compute_residual_momentum_signal(returns, factors, asof_mid, cfg)
    poisoned = returns.copy()
    poisoned.loc[poisoned.index > asof_mid] = 9.99
    signal_poisoned = compute_residual_momentum_signal(poisoned, factors, asof_mid, cfg)
    check("R1: asof 이후 수익률 오염에도 시그널 동일", signal_mid.equals(signal_poisoned))

    window = returns.loc[:asof].tail(cfg.est_window)
    f = factors.reindex(window.index)
    x = np.column_stack([np.ones(len(window)), f[list(FACTOR_COLS)].values])
    correlations = []
    for code in signal.nlargest(10).index:
        y = (window[code] - f["RF"]).values
        beta, *_ = np.linalg.lstsq(x, y.reshape(-1, 1), rcond=None)
        resid = y - (x @ beta).ravel()
        for factor in FACTOR_COLS:
            correlations.append(abs(np.corrcoef(resid, f[factor].values)[0, 1]))
    max_corr = float(np.nanmax(correlations))
    check("R2: 회귀 잔차와 팩터 상관 제거", max_corr < 0.10, f"max_corr={max_corr:.4f}")

    check("R3: stale 종목 제외", "S000" not in signal.index)

    bt = run_backtest(returns, factors, cfg)
    real_mean = float(bt.portfolio_returns.mean())
    rng = np.random.default_rng(123)
    placebo_means = []
    for _ in range(20):
        shuffled = returns.copy()
        for col in shuffled.columns:
            values = shuffled[col].values.copy()
            rng.shuffle(values)
            shuffled[col] = values
        bt_p = run_backtest(shuffled, factors, cfg)
        if not bt_p.portfolio_returns.empty:
            placebo_means.append(float(bt_p.portfolio_returns.mean()))
    percentile = float(np.mean([real_mean > value for value in placebo_means]))
    check("R4: 합성 데이터에서 time-shuffle placebo보다 우수", percentile >= 0.80, f"percentile={percentile:.3f}")

    corr_plain = float(bt.signal_corr_with_plain.mean())
    check("R5: 일반 12-1 모멘텀과 완전 동일하지 않음", corr_plain < 0.95, f"corr={corr_plain:.4f}")

    real_notes: dict[str, object] = {}
    try:
        real_returns, real_factors = load_real_inputs()
        real_cfg = ResMomConfig(est_window=36, top_n=20, min_obs=30)
        real_bt = run_backtest(real_returns, real_factors, real_cfg)
        real_notes = real_bt.to_payload()
        check("R6: 실제 수정주가 + FF 팩터 스모크", len(real_bt.portfolio_returns) >= 12)
    except Exception as exc:  # noqa: BLE001
        check("R6: 실제 수정주가 + FF 팩터 스모크", False, str(exc))

    out = {
        "status": "PASS" if not FAILURES else "FAIL",
        "synthetic": {
            "mean_monthly_return": round(real_mean, 6),
            "placebo_percentile": round(percentile, 3),
            "mean_rank_corr_with_plain": round(corr_plain, 4),
            "fp_json": bt.fp_json,
        },
        "real_smoke": real_notes,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print("-" * 72)
    if FAILURES:
        print(f"결과: FAIL ({len(FAILURES)}건)")
        return 1
    print(f"결과: ALL PASS - 저장: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

