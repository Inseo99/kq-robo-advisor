from __future__ import annotations

from kq_tool.backtest.strategy_meta import (
    KOSPI_BENCHMARK,
    QUANT,
    QUANT_ROBO_FILTER,
    QUANT_S2,
    QUANT_S2_ROBO_FILTER,
    normalize_strategy_key,
    strategy_descriptor,
)


def test_normalize_strategy_key_maps_aliases() -> None:
    assert normalize_strategy_key(None) == QUANT.key
    assert normalize_strategy_key("momentum") == QUANT.key
    assert normalize_strategy_key("s2_momentum") == QUANT_S2.key
    assert normalize_strategy_key("quant_on") == QUANT_ROBO_FILTER.key
    assert normalize_strategy_key("quant_s2_on") == QUANT_S2_ROBO_FILTER.key
    assert normalize_strategy_key("robo") == "robo"


def test_strategy_descriptor_returns_known_display_metadata() -> None:
    assert strategy_descriptor("quant_s2").label == "퀀트(S2모멘텀)"
    assert strategy_descriptor("s2_momentum").key == QUANT_S2.key
    assert KOSPI_BENCHMARK.label == "KOSPI"


def test_strategy_descriptor_falls_back_for_unknown_keys() -> None:
    desc = strategy_descriptor("custom")

    assert desc.key == "custom"
    assert desc.label == "custom"
