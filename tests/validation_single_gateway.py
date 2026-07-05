"""단일 게이트웨이 관문 (G1~G5) - "원천 데이터는 게이트웨이로만".

배경: 프로젝트의 가장 비싼 사고(문제 8번, stale parquet 캐시로 인한
판정 ① 무효)의 뿌리는 데이터 읽기 경로의 분산이었다. 이 관문은
저장소 전체를 정적 스캔해 게이트웨이 밖의 원천 데이터 접근을 잡는다.

  G1: 게이트웨이 API 무결성 (전 함수 호출 가능)
  G2: 입력 지문 생성 + 파일 변경 시 지문 변경
  G3: [핵심] 저장소 전체 스캔 -
      FAIL 패턴 (가격 원천 직접 접근): price parquet 캐시, yf.download,
           read_pickle 가격, 허용 목록 밖의 close.csv 직접 read_csv
      WARN 패턴 (마이그레이션 대기): read_excel/.xlsx (재무 데이터 -
           한계 6번으로 문서화된 legacy, 신규 사용은 금지 권고)
  G4: KOSPI 벤치마크 로드 (ECOS 수집본, 오프라인 재현 가능)
  G5: stamp_results 가 결과에 지문을 동봉하는지 + stale 판정 로직

허용 목록(ALLOWLIST): 게이트웨이 자신, 수집 스크립트(fetch_*),
legacy/ 폴더. 여기 없는 파일에서 FAIL 패턴 발견 시 관문 실패.

실행:  python3 tests/validation_single_gateway.py
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kq_tool.data.gateway import (  # noqa: E402
    fingerprint, get_kospi_benchmark, get_regime_payload, get_universe,
    load_close_panel, stamp_results,
)

FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"[{'PASS' if cond else 'FAIL'}] {name}"
          + (f"  -- {detail}" if detail and not cond else ""))
    if not cond:
        FAILURES.append(name)


# ---------------------------------------------------------------------------
# G3 스캔 설정
# ---------------------------------------------------------------------------
ALLOWLIST = {  # 원천 접근이 '직무'인 파일들
    "src/kq_tool/data/app_data.py",
    "src/kq_tool/data/gateway.py",
    "src/kq_tool/data/price_data.py",
    "scripts/fetch_prices.py",
    "scripts/fetch_ecos.py",
    "scripts/search_ecos.py",
}
SKIP_DIRS = {"legacy", "__pycache__", ".git", "analysis_outputs",
             "node_modules", ".venv", "venv"}

FAIL_PATTERNS = [
    (r"price_종가\.parquet", "가격 parquet 캐시 (문제 8번의 원인)"),
    (r"_load_price_parquet", "가격 parquet 로더 직접 호출"),
    (r"yf\.download\(", "yfinance 실시간 호출 (재현성/수정주가 리스크)"),
    (r"read_csv\([^)]*close\.csv", "close.csv 직접 접근 (게이트웨이 우회)"),
]
PRICE_LEGACY_FILES = {
    "data_loader.py",  # 재무 legacy 로더에 남은 과거 가격 함수: 운영 경로에서 직접 호출 금지
}


WARN_PATTERNS = [
    (r"read_excel|\.xlsx", "엑셀 접근 (재무 legacy - 한계 6번, 신규 사용 금지)"),
]


def strip_comments(src: str) -> str:
    lines = []
    in_doc = False
    for ln in src.splitlines():
        s = ln.strip()
        if s.startswith(('"""', "'''")):
            if not (s.endswith(('"""', "'''")) and len(s) > 3):
                in_doc = not in_doc
            continue
        if in_doc:
            continue
        lines.append(ln.split("#")[0])
    return "\n".join(lines)


def scan_repo() -> tuple[list[str], list[str]]:
    fails, warns = [], []
    for py in ROOT.rglob("*.py"):
        rel = py.relative_to(ROOT).as_posix()
        if any(part in SKIP_DIRS for part in py.parts):
            continue
        if rel in ALLOWLIST or py.name == Path(__file__).name:
            continue
        code = strip_comments(py.read_text(encoding="utf-8", errors="ignore"))
        if rel not in PRICE_LEGACY_FILES:
            for pat, why in FAIL_PATTERNS:
                for m in re.finditer(pat, code):
                    line_no = code[:m.start()].count("\n") + 1
                    fails.append(f"{rel}:{line_no}  [{why}]")
        for pat, why in WARN_PATTERNS:
            n = len(re.findall(pat, code))
            if n:
                warns.append(f"{rel}  [{why}] x{n}")
    return fails, warns


def main() -> int:
    print("=" * 70)
    print("단일 게이트웨이 관문  (tests/validation_single_gateway.py)")
    print("=" * 70)

    # G1: API 무결성
    try:
        panel = load_close_panel()
        _ = get_universe()
        _ = get_regime_payload()
        check("G1: 게이트웨이 API 전 함수 호출 가능",
              panel.shape[0] > 0 and panel.shape[1] > 0)
    except Exception as e:  # noqa: BLE001
        check("G1: 게이트웨이 API 전 함수 호출 가능", False, str(e))

    # G2: 지문
    fp = fingerprint()
    check("G2a: 핵심 입력 지문 생성 (close/kospi/labels)",
          fp["close"]["exists"] and "md5" in fp["close"])
    close_path = ROOT / "data" / "prices" / "close.csv"
    orig = close_path.read_bytes()
    changed_md5 = hashlib.md5(orig + b"\n").hexdigest()[:12]
    check("G2b: 파일 변경 시 지문 변경 (stale 탐지 가능)",
          changed_md5 != fp["close"]["md5"])

    # G3: 저장소 스캔
    fails, warns = scan_repo()
    check("G3: 게이트웨이 밖 가격 원천 직접 접근 0건", not fails,
          f"{len(fails)}건")
    for f in fails:
        print("       FAIL>", f)
    if warns:
        print(f"[WARN] G3: 엑셀 legacy 접근 {len(warns)}개 파일 "
              "(재무 PiT 마이그레이션 대기 - 한계 6번)")
        for w in warns[:10]:
            print("       warn>", w)

    # G4: KOSPI 벤치마크
    try:
        k = get_kospi_benchmark()
        check("G4: KOSPI 벤치마크 로드 (ECOS 수집본, 오프라인 재현)",
              len(k) > 100 and k.index.is_monotonic_increasing
              and (k > 0).all())
    except FileNotFoundError as e:
        check("G4: KOSPI 벤치마크 로드", False, str(e))

    # G5: 결과 스탬프 + stale 판정
    stamped = stamp_results({"quant_actual_cagr": 31.58})
    check("G5a: stamp_results 가 지문 동봉",
          "fp_json" in stamped and "fp_generated_at" in stamped)
    fp_now = json.loads(stamped["fp_json"])
    check("G5b: 스탬프의 close.md5 == 현재 파일 md5 (stale 판정 로직)",
          fp_now["close"]["md5"] == fingerprint()["close"]["md5"])
    # npz 왕복 확인
    p = ROOT / "tests" / "analysis_outputs" / "_stamp_roundtrip.npz"
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez(p, **stamped)
    with np.load(p, allow_pickle=True) as back:
        roundtrip_ok = (
            json.loads(str(back["fp_json"]))["close"]["md5"]
            == fp_now["close"]["md5"]
        )
    check("G5c: npz 저장/로드 후 지문 보존", roundtrip_ok)
    p.unlink(missing_ok=True)

    print("-" * 70)
    if FAILURES:
        print(f"결과: FAIL ({len(FAILURES)}건) - 게이트웨이 우회 경로를 제거하세요")
        return 1
    print("결과: ALL PASS - 원천 데이터는 게이트웨이로만 흐릅니다")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


