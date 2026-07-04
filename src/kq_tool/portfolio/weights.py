"""Portfolio weight normalization and combination helpers."""

from __future__ import annotations

from collections.abc import Iterable, Mapping


def normalize_weights(weights: Mapping[str, float]) -> dict[str, float]:
    """Normalize positive weights to sum to 1.0."""

    clean = {key: float(value) for key, value in weights.items() if value and value > 0}
    total = sum(clean.values())
    if total <= 0:
        return {}
    return {key: value / total for key, value in clean.items()}


def combine_weight_sets(
    weight_sets: Iterable[tuple[float, Mapping[str, float] | None]]
) -> dict[str, float]:
    """Blend multiple weight dictionaries using mixture weights."""

    combined: dict[str, float] = {}
    total_mix = 0.0
    for mix_weight, weights in weight_sets:
        if not weights:
            continue
        total_mix += float(mix_weight)
        for ticker, weight in weights.items():
            combined[ticker] = combined.get(ticker, 0.0) + float(mix_weight) * float(weight)
    if total_mix > 0:
        combined = {ticker: weight / total_mix for ticker, weight in combined.items()}
    return normalize_weights(combined)


def meta_base_weights(
    components: Iterable[Mapping[str, object]],
    strategies: Mapping[str, Mapping[str, float] | None],
) -> dict[str, float]:
    """Build the operating base portfolio from named strategy components."""

    sets = []
    for component in components:
        name = str(component["name"])
        weight = float(component["weight"])
        strategy_weights = strategies.get(name)
        if strategy_weights:
            sets.append((weight, strategy_weights))
    return combine_weight_sets(sets)


_normalize_weights = normalize_weights
_combine_weight_sets = combine_weight_sets
_meta_base_weights = meta_base_weights
