"""Validation segmentation helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence


def signal_names_for_side(
    signals: Mapping[str, tuple[int, str]],
    side: str = "all",
) -> list[str]:
    """Return signal names filtered by buy/sell direction."""

    side = side.lower()
    if side == "all":
        return list(signals.keys())
    if side not in {"buy", "sell"}:
        raise ValueError("side must be one of: all, buy, sell")
    wanted = 1 if side == "buy" else -1
    return [name for name, (direction, _) in signals.items() if direction == wanted]


def ticker_segment(tickers: Sequence[str], segment: str, split: int) -> list[str]:
    """Return large-cap or mid/small-cap slices from an already-ranked ticker list."""

    segment = segment.lower()
    if split <= 0:
        raise ValueError("split must be positive")
    if segment == "all":
        return list(tickers)
    if segment == "large":
        return list(tickers[:split])
    if segment in {"mid_small", "small", "rest"}:
        return list(tickers[split:])
    raise ValueError("segment must be one of: all, large, mid_small")


def segment_split_for_top(top_n: int, split: int | None = None) -> int:
    """Return a stable split point for a top-N ranked universe."""

    if top_n <= 1:
        return 1
    if split is None:
        return max(1, top_n // 2)
    return max(1, min(int(split), top_n - 1))
