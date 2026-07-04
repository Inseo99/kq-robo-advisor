from __future__ import annotations

from kq_tool.portfolio.weights import combine_weight_sets, meta_base_weights, normalize_weights


def test_normalize_weights_removes_non_positive_values() -> None:
    result = normalize_weights({"A": 2, "B": 0, "C": -1, "D": 2})

    assert result == {"A": 0.5, "D": 0.5}


def test_combine_weight_sets_blends_and_normalizes() -> None:
    result = combine_weight_sets(
        [
            (0.5, {"A": 1.0}),
            (0.5, {"B": 1.0}),
        ]
    )

    assert result == {"A": 0.5, "B": 0.5}


def test_combine_weight_sets_ignores_missing_strategy_fallbacks() -> None:
    result = combine_weight_sets(
        [
            (0.25, None),
            (0.75, {"A": 2.0, "B": 1.0}),
        ]
    )

    assert round(sum(result.values()), 10) == 1.0
    assert result == {"A": 2 / 3, "B": 1 / 3}


def test_meta_base_weights_uses_named_strategy_components() -> None:
    components = [{"name": "one", "weight": 0.25}, {"name": "two", "weight": 0.75}]
    strategies = {"one": {"A": 1.0}, "two": {"B": 1.0}}

    result = meta_base_weights(components, strategies)

    assert result == {"A": 0.25, "B": 0.75}


def test_meta_base_weights_remains_normalized_when_component_is_unavailable() -> None:
    components = [
        {"name": "missing", "weight": 0.40},
        {"name": "available", "weight": 0.60},
    ]
    strategies = {"available": {"A": 0.3, "B": 0.7}}

    result = meta_base_weights(components, strategies)

    assert round(sum(result.values()), 10) == 1.0
    assert result == {"A": 0.3, "B": 0.7}
