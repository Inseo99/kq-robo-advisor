"""Project-wide constants for KQ Quant Tool.

Only stable configuration lives here. Dynamic data such as the 3,000+
stock universe should be loaded through the data layer, not imported as
a giant module-level side effect.
"""

from __future__ import annotations

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
CACHE_DIR = DATA_DIR / "cache"

PORT = 8888
RISK_FREE_RATE = 0.035
EQUITY_RISK_PREMIUM = 0.060
SCREENER_LIMIT = 200

HORIZONS: list[int] = [1, 3, 5, 7, 10, 15, 20]

ETFS: dict[str, tuple[str, str, str]] = {
    "069500.KS": ("KODEX 200", "주식", "공격"),
    "229200.KS": ("KODEX 코스닥150", "성장주", "공격"),
    "130680.KS": ("TIGER 원유선물", "원자재", "공격"),
    "132030.KS": ("KODEX 골드선물", "금", "헤지"),
    "114260.KS": ("KODEX 국고채3년", "중기채", "수비"),
    "148070.KS": ("KOSEF 국고채10년", "장기채", "수비"),
    "153130.KS": ("KODEX 단기채권", "초단기", "수비"),
}

STRATEGIES: dict[str, dict[str, float] | None] = {
    "영구포트폴리오": {
        "069500.KS": 0.25,
        "132030.KS": 0.25,
        "148070.KS": 0.25,
        "153130.KS": 0.25,
    },
    "황금나비": {
        "069500.KS": 0.20,
        "229200.KS": 0.20,
        "132030.KS": 0.20,
        "148070.KS": 0.20,
        "153130.KS": 0.20,
    },
    "올웨더": {
        "069500.KS": 0.30,
        "130680.KS": 0.075,
        "132030.KS": 0.075,
        "148070.KS": 0.40,
        "114260.KS": 0.15,
    },
    "정적 60/40": {
        "069500.KS": 0.60,
        "148070.KS": 0.40,
    },
    "동일비중": {ticker: round(1 / 7, 4) for ticker in ETFS},
    "역변동성": None,
    "GMV": None,
    "MDP": None,
    "ERC": None,
    "GTAA": None,
}

RISK_BASED_KEYS: list[str] = ["GMV", "MDP", "ERC"]

# signal name -> (direction, robo sub-signal key)
# direction: +1 means a bullish edge, -1 means a bearish edge.
SIGNAL_DIRECTION: dict[str, tuple[int, str]] = {
    "RSI 과매도": (+1, "s_rsi"),
    "RSI 과매수": (-1, "s_rsi"),
    "MACD 골든크로스": (+1, "s_macd"),
    "MACD 데드크로스": (-1, "s_macd"),
    "BB 하단터치": (+1, "s_bb"),
    "BB 상단터치": (-1, "s_bb"),
}

ROBO_SIGNAL_WEIGHTS: dict[str, int] = {
    "s_rsi": 20,
    "s_macd": 25,
    "s_bb": 20,
    "s_ma20": 15,
    "s_ma60": 20,
}

ROBO_SIGNAL_LABELS: list[tuple[str, str]] = [
    ("s_rsi", "RSI(14)"),
    ("s_macd", "MACD"),
    ("s_bb", "볼린저밴드"),
    ("s_ma20", "MA20"),
    ("s_ma60", "MA60"),
]

ROBO_BUY_THRESHOLD = 65
ROBO_SELL_THRESHOLD = 35

REGIMES: list[str] = ["골디락스", "리플레이션", "스태그플레이션", "디플레이션"]
