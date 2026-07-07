"""단일 데이터 게이트웨이 — app_data 확장 (벤치마크 + 입력 지문).

원칙
----
"모든 원천 데이터는 이 모듈(과 app_data)을 통해서만 읽는다."
프로젝트에서 겪은 사고(분할 미수정 가격, 상폐 잔존, 검증-앱 불일치,
stale parquet 캐시)의 공통 뿌리는 데이터 읽기 경로의 분산이었다.
tests/validation_single_gateway.py 가 이 원칙을 저장소 전체에 정적으로 강제한다.

추가 제공
---------
1. get_kospi_benchmark(): ECOS 수집본(data/macro/kospi.csv) 기반 벤치마크.
   validation_core 의 yf.download('^KS11') 실시간 호출을 대체 —
   네트워크 실패 시 price_df.mean() 으로 조용히 벤치마크가 바뀌는
   재현성 구멍을 제거한다.
2. fingerprint(): 핵심 입력 파일들의 해시/마지막 관측일.
3. stamp_results(save_data): 검증 결과 npz 에 입력 지문을 동봉 —
   "재검증했다"는 보고와 "실제로 새 데이터를 읽었다"는 사실이
   자동으로 대조된다 (문제 8번의 구조적 재발 방지).

사용법 (검증 스크립트에서):
    from kq_tool.data.gateway import get_kospi_benchmark, stamp_results
    ...
    np.savez(save_path, **stamp_results(save_data))
"""

from __future__ import annotations

import getpass
import hashlib
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

# app_data 의 기존 게이트웨이 함수 재수출 — import 지점을 한 곳으로
from .app_data import (  # noqa: F401
    get_macro_snapshot, get_regime_payload, get_universe, load_close_panel,
)

ROOT = Path(__file__).resolve().parents[3]

__all__ = ["load_close_panel", "get_universe", "get_macro_snapshot",
           "get_regime_payload", "get_kospi_benchmark",
           "fingerprint", "stamp_results"]

_CORE_INPUTS = {
    "close": ROOT / "data" / "prices" / "close.csv",
    "kospi": ROOT / "data" / "macro" / "kospi.csv",
    "gdp_qoq": ROOT / "data" / "macro" / "gdp_qoq.csv",
    "labels": ROOT / "data" / "macro" / "regime_labels.csv",
}


def get_kospi_benchmark(start: str | None = None) -> pd.Series:
    """KOSPI 지수 (ECOS 수집본) — 벤치마크 용도.

    yf.download 실시간 호출 대체: 오프라인 재현 가능, vintage 고정.
    """
    f = _CORE_INPUTS["kospi"]
    if not f.exists():
        raise FileNotFoundError(
            f"{f} 없음 — scripts/fetch_ecos.py 실행 필요 "
            "(yf.download 폴백은 재현성 문제로 제공하지 않음)")
    s = pd.read_csv(f, parse_dates=["date"]).set_index("date")["value"]
    s = s.sort_index().dropna()
    return s.loc[start:] if start else s


def _file_fp(path: Path) -> dict:
    if not path.exists():
        return {"exists": False}
    h = hashlib.md5(path.read_bytes()).hexdigest()[:12]
    out = {"exists": True, "md5": h, "bytes": path.stat().st_size}
    try:  # 날짜 컬럼이 있는 csv 면 마지막 관측일도
        tail = pd.read_csv(path, usecols=[0]).iloc[-1, 0]
        out["last_row"] = str(tail)
    except Exception:  # noqa: BLE001
        pass
    return out


def fingerprint() -> dict:
    """핵심 입력 파일들의 지문. 결과 해석/감사의 기준점."""
    return {name: _file_fp(p) for name, p in _CORE_INPUTS.items()}


def stamp_results(save_data: dict) -> dict:
    """검증 결과 dict 에 입력 지문을 동봉 (np.savez 호환: 문자열로 직렬화).

    검증 규칙: 결과를 읽을 때 fp_json 의 close.md5 가 현재
    data/prices/close.csv 의 md5 와 다르면 그 결과는 stale 이다.
    """
    save_data = dict(save_data)
    save_data["fp_json"] = json.dumps(fingerprint(), ensure_ascii=False)
    save_data["fp_generated_at"] = datetime.now().isoformat(timespec="seconds")
    save_data["fp_generated_by"] = getpass.getuser()
    return save_data
