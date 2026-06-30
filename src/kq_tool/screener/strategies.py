"""Screener ranking rules."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

SCREENER_DEFINITIONS = {
    "저PER": {"key": "pe", "reverse": False, "allow_nonpositive": False},
    "저PBR": {"key": "pbr", "reverse": False, "allow_nonpositive": False},
    "고ROE": {"key": "roe", "reverse": True, "allow_nonpositive": False},
    "모멘텀": {"key": "mom", "reverse": True, "allow_nonpositive": False},
    "역방향DCF": {"key": "dcf_g", "reverse": False, "allow_nonpositive": True},
    "S2모멘텀": {"key": "s2_mom", "reverse": True, "allow_nonpositive": False},
    "로보매수": {"key": "score", "reverse": True, "allow_nonpositive": False},
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


def build_screeners(
    results: Mapping[str, Mapping[str, object]],
    limit: int = 8,
) -> dict[str, list[str]]:
    """Build all configured screener ticker lists."""

    return {
        name: rank_by(
            results,
            key=str(config["key"]),
            reverse=bool(config["reverse"]),
            allow_nonpositive=bool(config["allow_nonpositive"]),
            limit=limit,
        )
        for name, config in SCREENER_DEFINITIONS.items()
    }
