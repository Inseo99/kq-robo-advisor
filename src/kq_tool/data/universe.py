"""Universe construction helpers."""

from __future__ import annotations

from collections.abc import Mapping

FallbackUniverse = dict[str, tuple[str, str]]

FALLBACK_UNIVERSE: FallbackUniverse = {
    "005930.KS": ("삼성전자", "IT/반도체"),
    "000660.KS": ("SK하이닉스", "IT/반도체"),
    "373220.KS": ("LG에너지솔루션", "2차전지"),
    "207940.KS": ("삼성바이오로직스", "바이오"),
    "005380.KS": ("현대차", "자동차"),
    "000270.KS": ("기아", "자동차"),
    "051910.KS": ("LG화학", "화학/2차전지"),
    "006400.KS": ("삼성SDI", "2차전지"),
    "035420.KS": ("NAVER", "IT/플랫폼"),
    "035720.KS": ("카카오", "IT/플랫폼"),
    "068270.KS": ("셀트리온", "바이오"),
    "105560.KS": ("KB금융", "금융"),
    "055550.KS": ("신한지주", "금융"),
    "086790.KS": ("하나금융지주", "금융"),
    "066570.KS": ("LG전자", "전자"),
    "005490.KS": ("POSCO홀딩스", "철강"),
    "015760.KS": ("한국전력", "유틸리티"),
    "030200.KS": ("KT&G", "통신"),
    "096770.KS": ("SK이노베이션", "에너지/화학"),
    "003550.KS": ("LG", "지주"),
}


def yahoo_suffix(market: str | None) -> str:
    """Return Yahoo Finance suffix for a Korean market label."""

    return ".KS" if market == "KOSPI" else ".KQ"


def build_universe(excel_data: Mapping[str, Mapping[str, object]] | None) -> FallbackUniverse:
    """Build Yahoo-style ticker universe from loaded stock metadata."""

    if not excel_data:
        return dict(FALLBACK_UNIVERSE)

    universe: FallbackUniverse = {}
    for ticker, info in excel_data.items():
        market = str(info.get("market") or "")
        suffix = yahoo_suffix(market)
        yahoo_ticker = f"{ticker}{suffix}"
        name = str(info.get("name") or ticker)
        sector = str(info.get("sector") or "기타")
        universe[yahoo_ticker] = (name, sector)
    return universe


def market_counts(universe: Mapping[str, object]) -> dict[str, int]:
    """Count KOSPI/KOSDAQ-style ticker suffixes in a universe."""

    return {
        "KOSPI": sum(1 for ticker in universe if ticker.endswith(".KS")),
        "KOSDAQ": sum(1 for ticker in universe if ticker.endswith(".KQ")),
    }
