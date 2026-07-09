"""Run the core validation gates and write a dashboard JSON.

The dashboard deliberately records ``data_source`` and ``validation_stage`` so
synthetic self-checks cannot be mistaken for real-data empirical validation.

Usage:
    python scripts\run_all_gates.py
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tests" / "analysis_outputs" / "gates_dashboard.json"

DEFAULT_GATES = [
    ("macro_pit", "tests/validation_macro_pit.py", "real"),
    ("price_integrity", "tests/validation_price_integrity.py", "real"),
    ("financial_pit", "tests/validation_financial_pit.py", "real"),
    ("single_gateway", "tests/validation_single_gateway.py", "real"),
    ("app_wiring", "tests/validation_app_wiring.py", "real"),
    ("transition_pit", "tests/validation_transition_pit.py", "real"),
    ("label_vintage", "tests/validation_label_vintage.py", "real"),
    ("tea_band", "tests/validation_tea_band.py", "real"),
    ("regime_erc_v1", "tests/validation_regime_erc_v1.py", "real"),
]


def _run_gate(name: str, script: str, data_source: str, timeout: int) -> dict[str, object]:
    cmd = [sys.executable, script]
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
    )
    output = proc.stdout or ""
    return {
        "name": name,
        "script": script,
        "data_source": data_source,
        "validation_stage": "empirical_real_data" if data_source == "real" else "synthetic_self_check",
        "status": "PASS" if proc.returncode == 0 else "FAIL",
        "returncode": int(proc.returncode),
        "summary_line": next(
            (line.strip() for line in reversed(output.splitlines()) if line.strip()),
            "",
        ),
        "stdout_tail": output.splitlines()[-25:],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--continue-on-fail", action="store_true")
    args = parser.parse_args()

    results = []
    overall = "PASS"
    for name, script, data_source in DEFAULT_GATES:
        print("=" * 72)
        print(f"[gate] {name} ({data_source})")
        print("=" * 72)
        result = _run_gate(name, script, data_source, args.timeout)
        results.append(result)
        print(result["summary_line"])
        if result["status"] != "PASS":
            overall = "FAIL"
            if not args.continue_on_fail:
                break

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "overall_status": overall,
        "validation_stage": "empirical_real_data",
        "data_source": "real",
        "root": str(ROOT),
        "gates": results,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("=" * 72)
    print(f"dashboard saved: {OUT}")
    print(f"overall_status: {overall}")
    return 0 if overall == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
