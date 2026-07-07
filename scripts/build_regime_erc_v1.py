"""Build regime base allocation v1 with kappa-shrinkage ERC.

Output:
  data/analysis_outputs/regime_erc_v1_weights.csv
  data/analysis_outputs/regime_erc_v1_audit.csv
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kq_tool.portfolio.regime_erc import RegimeERCConfig, build_regime_erc_allocations  # noqa: E402
from kq_tool.regime.transition_pit import load_regime_history  # noqa: E402


def load_returns() -> pd.DataFrame:
    path = ROOT / "data" / "assets" / "monthly_returns.csv"
    df = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
    df.index = df.index.to_period("M").to_timestamp("M")
    return df


def main() -> int:
    returns = load_returns()
    labels = load_regime_history()
    cfg = RegimeERCConfig(n0=36.0)
    weights, audit = build_regime_erc_allocations(returns, labels, cfg)

    out_dir = ROOT / "data" / "analysis_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    weights_path = out_dir / "regime_erc_v1_weights.csv"
    audit_path = out_dir / "regime_erc_v1_audit.csv"
    meta_path = out_dir / "regime_erc_v1_meta.json"
    weights.to_csv(weights_path, encoding="utf-8-sig")
    audit.to_csv(audit_path, index=False, encoding="utf-8-sig")
    meta_path.write_text(
        json.dumps(
            {
                "method": "kappa-shrinkage bounded ERC",
                "data_source": "real ETF monthly returns + real PiT regime labels",
                "returns_file": "data/assets/monthly_returns.csv",
                "labels_file": "data/regime/labels.csv",
                "n0": cfg.n0,
                "note": "v1 uses real ETF sample only; no pre-listing index proxies.",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("=" * 72)
    print("Regime ERC v1 weights")
    print("=" * 72)
    print((weights * 100).round(1).to_string())
    print()
    print("Audit")
    print(audit[["regime", "n_months", "kappa", "max_weight", "min_weight"]].round(4).to_string(index=False))
    print(f"saved: {weights_path}")
    print(f"saved: {audit_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
