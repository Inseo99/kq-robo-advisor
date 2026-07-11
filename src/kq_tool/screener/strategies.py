"""Screener ranking rules for the six independent backtest strategies."""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np

DISPLAY_NAME_QUALITY = "중형성장주"

SCREENER_DEFINITIONS = {
    "s1m_lsv": {
        "label": "M1 LSV",
        "selection": "KR 섹터 Fama-LSV",
        "gate": "없음(상시 투자)",
        "ranker": "lsv",
        "note": "PER·PBR이 모두 양수인 종목을 듀얼 밸류 순위로 정렬합니다.",
    },
    "s2m_lsv": {
        "label": "M2 LSV",
        "selection": "KR 섹터 Fama-LSV",
        "gate": "t1: 미국 위험신호 3개 중 2개 이상이면 현금",
        "ranker": "lsv",
        "note": "종목선정은 s1m_lsv와 같고, 백테스트에서는 t1 위험회피 게이트만 추가됩니다.",
    },
    "s3m_lsv": {
        "label": "M3 LSV",
        "selection": "KR 섹터 Fama-LSV",
        "gate": "t2: KODEX200 10개월 이동평균 하회 시 현금",
        "ranker": "lsv",
        "note": "종목선정은 s1m_lsv와 같고, 백테스트에서는 t2 추세 게이트만 추가됩니다.",
    },
    "s1m_kang": {
        "label": f"M1 {DISPLAY_NAME_QUALITY}",
        "selection": f"KR 섹터 {DISPLAY_NAME_QUALITY}",
        "gate": "없음(상시 투자)",
        "ranker": "kang",
        "note": "시총 하위 50%, 영업현금흐름·순이익 양수, 영업이익/자산 수익성 높은 순서로 정렬합니다. 정식 필드가 없으면 ROE 기반 fallback을 명시 표시합니다.",
    },
    "s2m_kang": {
        "label": f"M2 {DISPLAY_NAME_QUALITY}",
        "selection": f"KR 섹터 {DISPLAY_NAME_QUALITY}",
        "gate": "t1: 미국 위험신호 3개 중 2개 이상이면 현금",
        "ranker": "kang",
        "note": "종목선정은 s1m_kang과 같고, 백테스트에서는 t1 위험회피 게이트만 추가됩니다.",
    },
    "s3m_kang": {
        "label": f"M3 {DISPLAY_NAME_QUALITY}",
        "selection": f"KR 섹터 {DISPLAY_NAME_QUALITY}",
        "gate": "t2: KODEX200 10개월 이동평균 하회 시 현금",
        "ranker": "kang",
        "note": "종목선정은 s1m_kang과 같고, 백테스트에서는 t2 추세 게이트만 추가됩니다.",
    },
}


def is_rankable_value(value: object, reverse: bool = False, allow_nonpositive: bool = False) -> bool:
    """Return whether a screener value should participate in ranking."""

    if value is None:
        return False
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    if np.isnan(numeric):
        return False
    if allow_nonpositive or reverse:
        return True
    return numeric > 0


def rank_by(
    results: Mapping[str, Mapping[str, object]],
    key: str,
    reverse: bool = False,
    allow_nonpositive: bool = False,
    limit: int = 8,
) -> list[str]:
    """Rank tickers by one screener field."""

    pairs = [
        (ticker, float(data[key]))
        for ticker, data in results.items()
        if key in data and is_rankable_value(data.get(key), reverse, allow_nonpositive)
    ]
    pairs.sort(key=lambda item: item[1], reverse=reverse)
    return [ticker for ticker, _ in pairs[:limit]]


def _numeric(data: Mapping[str, object], key: str) -> float | None:
    try:
        value = float(data.get(key) or 0)
    except (TypeError, ValueError):
        return None
    if np.isnan(value):
        return None
    return value


def _has_yfinance_kang_fields(data: Mapping[str, object]) -> bool:
    """Return whether all formal Kang fields from the yfinance cache exist."""

    return all(
        value is not None and value > 0
        for value in (
            _numeric(data, "operating_cashflow"),
            _numeric(data, "net_income_common"),
            _numeric(data, "total_assets"),
            _numeric(data, "operating_income"),
        )
    )


def _rank_lsv(results: Mapping[str, Mapping[str, object]], limit: int) -> list[str]:
    valid: list[tuple[str, float, float]] = []
    for ticker, data in results.items():
        pe = _numeric(data, "pe")
        pbr = _numeric(data, "pbr")
        if pe is None or pbr is None or pe <= 0 or pbr <= 0:
            continue
        valid.append((ticker, pe, pbr))

    pe_rank = {ticker: i + 1 for i, (ticker, _, _) in enumerate(sorted(valid, key=lambda x: (x[1], x[0])))}
    pbr_rank = {ticker: i + 1 for i, (ticker, _, _) in enumerate(sorted(valid, key=lambda x: (x[2], x[0])))}
    scored = [
        (ticker, pe_rank[ticker] + pbr_rank[ticker], pe, pbr)
        for ticker, pe, pbr in valid
    ]
    scored.sort(key=lambda x: (x[1], x[2], x[3], x[0]))
    return [ticker for ticker, *_ in scored[:limit]]


def _rank_kang(results: Mapping[str, Mapping[str, object]], limit: int) -> list[str]:
    strict: list[tuple[str, float, float]] = []
    fallback: list[tuple[str, float, float]] = []
    for ticker, data in results.items():
        market_cap = _numeric(data, "mcap")
        quality = _numeric(data, "quality")
        if market_cap is None or market_cap <= 0 or quality is None:
            continue
        if _has_yfinance_kang_fields(data):
            strict.append((ticker, market_cap, quality))
        elif quality > 0:
            fallback.append((ticker, market_cap, quality))

    valid = strict if strict else fallback
    if not valid:
        return []
    valid.sort(key=lambda x: (x[1], x[0]))
    small_count = max(1, int(math.ceil(len(valid) * 0.5)))
    small = valid[:small_count]
    small.sort(key=lambda x: (-x[2], x[1], x[0]))
    return [ticker for ticker, *_ in small[:limit]]


def kang_mode(results: Mapping[str, Mapping[str, object]]) -> str:
    """Return whether Kang can use its strict yfinance accounting fields."""

    has_strict = False
    has_fallback = False
    for data in results.values():
        market_cap = _numeric(data, "mcap")
        quality = _numeric(data, "quality")
        if market_cap is None or market_cap <= 0 or quality is None:
            continue
        if _has_yfinance_kang_fields(data):
            has_strict = True
        elif quality > 0:
            has_fallback = True
    if has_strict:
        return "strict_yfinance"
    if has_fallback:
        return "fallback_roe"
    return "empty"


def screener_diagnostics(
    results: Mapping[str, Mapping[str, object]],
    screeners: Mapping[str, list[str]],
) -> dict[str, object]:
    """Build funnel diagnostics for the screener UI/API."""

    lsv_positive = 0
    kang_strict = 0
    kang_fallback = 0
    sector_mapped = 0
    for data in results.values():
        pe = _numeric(data, "pe")
        pbr = _numeric(data, "pbr")
        if pe is not None and pe > 0 and pbr is not None and pbr > 0:
            lsv_positive += 1
        market_cap = _numeric(data, "mcap")
        quality = _numeric(data, "quality")
        if market_cap is not None and market_cap > 0 and quality is not None:
            kang_fallback += int(quality > 0)
            if _has_yfinance_kang_fields(data):
                kang_strict += 1
        if data.get("sector"):
            sector_mapped += 1

    mode = kang_mode(results)
    active_pool = kang_strict if mode == "strict_yfinance" else kang_fallback if mode == "fallback_roe" else 0
    small_cap_bucket = int(math.ceil(active_pool * 0.5)) if active_pool else 0
    kang_final_selected = max(
        (len(value) for key, value in screeners.items() if "kang" in key),
        default=0,
    )
    warnings: list[str] = []
    if mode == "fallback_roe":
        warnings.append(
            f"{DISPLAY_NAME_QUALITY} 정식 재무필드(영업현금흐름·순이익)가 부족해 화면 후보는 시총하위50%+ROE 기준 fallback입니다."
        )
    elif mode == "empty":
        warnings.append(
            f"{DISPLAY_NAME_QUALITY} 후보 산출에 필요한 시총·수익성 필드가 부족합니다.")

    return {
        "us_sectors_loaded": 0,
        "top3_sectors": [],
        "sector_mapped_count": sector_mapped,
        "per_filter_count": {
            "lsv_positive_pe_pbr": lsv_positive,
            "kang_strict_yfinance": kang_strict,
            "kang_strict_cashflow_income": kang_strict,
            "kang_fallback_positive_roe": kang_fallback,
            "kang_active_pool": active_pool,
            "kang_small_cap_bucket": small_cap_bucket,
            "kang_final_selected": kang_final_selected,
            **{key: len(value) for key, value in screeners.items()},
        },
        "kang_mode": mode,
        "warnings": warnings,
        "note": (
            "현재 화면은 6개 백테스트 전략의 현재 후보를 보여주는 읽기 전용 스크리너입니다. "
            "미국 섹터 ETF 모멘텀은 전략검증 백테스트의 상류 신호이며, 이 화면에서는 후보 진단값으로만 분리 표시합니다."
        ),
    }


def build_screeners(
    results: Mapping[str, Mapping[str, object]],
    limit: int = 8,
) -> dict[str, list[str]]:
    """Build all configured screener ticker lists."""

    out: dict[str, list[str]] = {}
    for name, config in SCREENER_DEFINITIONS.items():
        ranker = str(config["ranker"])
        if ranker == "lsv":
            out[name] = _rank_lsv(results, limit)
        elif ranker == "kang":
            out[name] = _rank_kang(results, limit)
        else:
            out[name] = []
    return out


def screener_metadata() -> list[dict[str, str]]:
    """Return display metadata for the frontend tabs."""

    return [
        {
            "key": key,
            "label": str(config["label"]),
            "selection": str(config["selection"]),
            "gate": str(config["gate"]),
            "note": str(config["note"]),
        }
        for key, config in SCREENER_DEFINITIONS.items()
    ]


