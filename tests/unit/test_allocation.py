from __future__ import annotations

import numpy as np
import pandas as pd

from kq_tool.portfolio.allocation import (
    evaluate_asset_allocation_strategies,
    gtaa_weights,
    inverse_volatility_weights,
    risk_based_strategy_weights,
)


def _monthly_returns() -> pd.DataFrame:
    index = pd.date_range("2024-01-31", periods=12, freq="ME")
    return pd.DataFrame(
        {
            "A": [0.01, 0.02, -0.01, 0.03, 0.01, 0.00, 0.02, 0.01, -0.02, 0.03, 0.02, 0.01],
            "B": [0.00, 0.01, 0.01, -0.01, 0.02, 0.01, 0.00, 0.01, 0.01, -0.01, 0.02, 0.01],
            "C": [-0.01, 0.00, 0.02, 0.01, -0.01, 0.02, 0.01, 0.00, 0.01, 0.02, -0.01, 0.00],
        },
        index=index,
    )


def test_evaluate_asset_allocation_strategies_returns_legacy_payload_shape() -> None:
    result = evaluate_asset_allocation_strategies(
        _monthly_returns(),
        {
            "혼합": {"A": 0.6, "B": 0.4},
            "비어있음": None,
            "없는종목": {"Z": 1.0},
        },
        risk_based_keys=("GMV",),
        risk_free_rate=0.0,
    )

    assert set(result) == {"혼합"}
    payload = result["혼합"]
    assert payload["weights"] == {"A": 0.6, "B": 0.4}
    assert len(payload["cum"]) == 12
    assert len(payload["dates"]) == 12
    assert set(payload["metrics"]) == {"cagr", "vol", "sharpe", "mdd", "calmar"}
    assert set(payload["risk_contrib"]) == {"A", "B"}
    assert np.isclose(sum(payload["capital_weight"].values()), 100.0)
    assert payload["diversification_ratio"] > 0
    assert payload["is_risk_based"] is False


def test_evaluate_asset_allocation_strategies_marks_risk_based_keys() -> None:
    result = evaluate_asset_allocation_strategies(
        _monthly_returns(),
        {"GMV": {"A": 0.5, "B": 0.5}},
        risk_based_keys=("GMV",),
    )

    assert result["GMV"]["is_risk_based"] is True


def test_inverse_volatility_weights_sum_to_one() -> None:
    prices = {
        "A": pd.DataFrame({"Close": np.linspace(100, 120, 90)}),
        "B": pd.DataFrame({"Close": np.linspace(100, 110, 90)}),
    }

    weights = inverse_volatility_weights(prices, tickers=("A", "B"))

    assert set(weights) == {"A", "B"}
    assert np.isclose(sum(weights.values()), 1.0)


def test_gtaa_weights_uses_fallback_when_no_asset_is_above_average() -> None:
    prices = {
        "A": pd.DataFrame({"Close": np.linspace(120, 100, 220)}),
        "B": pd.DataFrame({"Close": np.linspace(110, 90, 220)}),
        "C": pd.DataFrame({"Close": np.linspace(105, 95, 220)}),
    }

    weights = gtaa_weights(prices, tickers=("A", "B", "C"), fallback_count=2)

    assert weights == {"A": 0.5, "B": 0.5}


def test_risk_based_strategy_weights_returns_three_dynamic_strategies() -> None:
    result = risk_based_strategy_weights(
        _monthly_returns(),
        gmv_fn=lambda cov: np.ones(cov.shape[0]) / cov.shape[0],
        mdp_fn=lambda cov, vols: np.ones(cov.shape[0]) / cov.shape[0],
        erc_fn=lambda cov: np.ones(cov.shape[0]) / cov.shape[0],
    )

    assert set(result) == {"GMV", "MDP", "ERC"}
    assert all(np.isclose(sum(weights.values()), 0.9999) or np.isclose(sum(weights.values()), 1.0)
               for weights in result.values() if weights)
