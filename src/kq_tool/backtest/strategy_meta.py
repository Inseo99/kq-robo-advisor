"""Backtest strategy keys, labels, and display metadata."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StrategyDescriptor:
    key: str
    label: str
    color: str


QUANT = StrategyDescriptor("quant", "퀀트(모멘텀)", "#388bfd")
QUANT_S2 = StrategyDescriptor("quant_s2", "퀀트(S2모멘텀)", "#3fb950")
QUANT_COMPARE = StrategyDescriptor("quant_compare", "퀀트 비교", "#388bfd")
ROBO = StrategyDescriptor("robo", "로보신호", "#bc8cff")
KOSPI_BENCHMARK = StrategyDescriptor("kospi", "KOSPI", "#8b949e")

STRATEGY_ALIASES = {
    "momentum": QUANT.key,
    "s2_momentum": QUANT_S2.key,
}

STRATEGY_DESCRIPTORS = {
    item.key: item
    for item in (
        QUANT,
        QUANT_S2,
        QUANT_COMPARE,
        ROBO,
        KOSPI_BENCHMARK,
    )
}


def normalize_strategy_key(strategy: str | None) -> str:
    """Return the canonical strategy key used by backtest internals."""

    key = (strategy or QUANT.key).strip()
    return STRATEGY_ALIASES.get(key, key)


def strategy_descriptor(strategy: str | None) -> StrategyDescriptor:
    """Return display metadata for a strategy, falling back to the raw key."""

    key = normalize_strategy_key(strategy)
    return STRATEGY_DESCRIPTORS.get(key, StrategyDescriptor(key, key, "#8b949e"))
