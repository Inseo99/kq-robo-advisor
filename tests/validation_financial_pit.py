"""Financial Point-in-Time validation gate.

This gate closes the previous Excel-financial WARN path.  Raw Excel parsing is
still allowed in data_loader.py, but every consumer must use the PiT helper that
shifts fiscal period rows to their observable/publication dates.

Run:
    python tests\validation_financial_pit.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from kq_tool.data.financial_pit import (  # noqa: E402
    MCAP_KEY,
    available_financial_rows,
    build_mcap_history_pit,
    financial_publication_lag_days,
    latest_financials_asof,
    latest_mcap_tickers_asof,
    normalize_financial_frame,
    observable_date_for_period,
)

FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  -- {detail}" if detail and not cond else ""))
    if not cond:
        FAILURES.append(name)


def _synthetic_fin() -> pd.DataFrame:
    rows = [
        ("005930", MCAP_KEY, "2024-03-31", 100.0),
        ("005930", "당기순이익(천원)", "2024-03-31", 10.0),
        ("005930", MCAP_KEY, "2024-06-30", 200.0),
        ("005930", "당기순이익(천원)", "2024-06-30", 20.0),
        ("005930", MCAP_KEY, "2024-12-31", 400.0),
        ("005930", "당기순이익(천원)", "2024-12-31", 40.0),
        ("000660", MCAP_KEY, "2024-03-31", 300.0),
    ]
    return pd.DataFrame(rows, columns=["ticker", "아이템명", "date", "value"])


def _scan_direct_legacy_calls() -> list[str]:
    offenders: list[str] = []
    allowed = {
        "data_loader.py",
        "tests/validation_financial_pit.py",
    }
    pattern = re.compile(r"(?<!def )get_latest_fin\(")
    for path in ROOT.rglob("*.py"):
        rel = path.relative_to(ROOT).as_posix()
        if any(part in {"legacy", ".git", "__pycache__", ".venv", "venv"} for part in path.parts):
            continue
        if rel in allowed or path.name in allowed:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for match in pattern.finditer(text):
            line_no = text[: match.start()].count("\n") + 1
            offenders.append(f"{rel}:{line_no}")
    return offenders


def main() -> int:
    print("=" * 72)
    print("Financial PiT validation  (tests/validation_financial_pit.py)")
    print("=" * 72)

    fin = _synthetic_fin()
    normalized = normalize_financial_frame(fin)

    check(
        "F1: Q1~Q3 lag = 45일, Q4 lag = 90일",
        financial_publication_lag_days("2024-03-31") == 45
        and financial_publication_lag_days("2024-12-31") == 90
        and observable_date_for_period("2024-03-31") == pd.Timestamp("2024-05-15")
        and observable_date_for_period("2024-12-31") == pd.Timestamp("2025-03-31"),
    )

    before = available_financial_rows(fin, ticker="005930", item=MCAP_KEY, asof="2024-05-14")
    after = available_financial_rows(fin, ticker="005930", item=MCAP_KEY, asof="2024-05-15")
    check(
        "F2: 발표 전 분기 재무는 보이지 않고 발표일 이후만 보임",
        before.empty and len(after) == 1 and float(after.iloc[-1]["value"]) == 100.0,
    )

    latest_early = latest_financials_asof(fin, "005930", asof="2024-08-13")
    latest_late = latest_financials_asof(fin, "005930", asof="2024-08-14")
    check(
        "F3: asof 변경에 따라 trailing 재무가 결정적으로 변함",
        latest_early.get(MCAP_KEY) == 100.0 and latest_late.get(MCAP_KEY) == 150.0,
        f"early={latest_early.get(MCAP_KEY)}, late={latest_late.get(MCAP_KEY)}",
    )

    hist = build_mcap_history_pit(
        {"005930.KS": ("삼성전자", "IT"), "000660.KS": ("SK하이닉스", "IT")},
        lambda ticker: ticker.split(".")[0],
        fin,
    )
    check(
        "F4: 시총 히스토리 인덱스는 회계기간말이 아니라 observable_date",
        pd.Timestamp("2024-03-31") not in hist.index
        and pd.Timestamp("2024-05-15") in hist.index
        and float(hist.loc[pd.Timestamp("2024-05-15"), "005930.KS"]) == 100.0,
    )

    top = latest_mcap_tickers_asof(fin, asof="2024-05-15", top_n=2)
    check("F5: 시총 랭킹도 asof 기준으로만 산출", top == ["000660", "005930"], str(top))

    offenders = _scan_direct_legacy_calls()
    check("F6: 운영/검증 코드에서 get_latest_fin 직접 호출 0건", not offenders, ", ".join(offenders[:5]))

    # Real-data smoke: if the user's Excel cache exists, ensure no row later than
    # asof leaks into an old snapshot.
    try:
        import data_loader

        real = data_loader.load_financials()
        rows = available_financial_rows(real, ticker="005930", asof="2018-01-31")
        check(
            "F7: 실제 엑셀 재무도 asof 필터 적용 가능",
            rows.empty or rows["observable_date"].max() <= pd.Timestamp("2018-01-31"),
        )
    except Exception as exc:  # noqa: BLE001
        check("F7: 실제 엑셀 재무도 asof 필터 적용 가능", False, str(exc))

    # Keep variable referenced so import/normalization failures are explicit.
    check("F8: 정규화 결과 필수 컬럼 보유", {"period_date", "observable_date"}.issubset(normalized.columns))

    print("-" * 72)
    if FAILURES:
        print(f"결과: FAIL ({len(FAILURES)}건)")
        return 1
    print("결과: ALL PASS - 엑셀 재무 데이터는 PiT 공시지연 기준으로 흐릅니다")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

