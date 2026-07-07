from __future__ import annotations

import numpy as np
import pandas as pd

from kq_tool.validation.factor_analysis import (
    FF3_FACTORS,
    cross_sectional_factor_returns,
    factor_regression_rows,
    fama_french_regression,
    run_factor_regressions,
)


def _factor_frame(n: int = 80) -> pd.DataFrame:
    index = pd.date_range("2020-01-31", periods=n, freq="ME")
    t = np.arange(n, dtype=float)
    return pd.DataFrame(
        {
            "MKT": 0.004 + np.sin(t / 5) * 0.01,
            "SMB": np.cos(t / 7) * 0.006,
            "HML": np.sin(t / 11) * 0.005,
            "RMW": np.cos(t / 13) * 0.004,
            "CMA": np.sin(t / 17) * 0.003,
        },
        index=index,
    )


def test_fama_french_regression_recovers_alpha_and_betas() -> None:
    factors = _factor_frame()
    portfolio = (
        0.001
        + 1.2 * factors["MKT"]
        + 0.4 * factors["SMB"]
        - 0.2 * factors["HML"]
        + 0.3 * factors["RMW"]
        + 0.1 * factors["CMA"]
    )

    result = fama_french_regression(portfolio, factors, model="ff5")

    assert result["ok"] is True
    assert round(result["alpha_period_pct"], 3) == 0.1
    assert round(result["alpha_annual_pct"], 2) == 1.2
    assert round(result["betas"]["MKT"], 2) == 1.2
    assert round(result["betas"]["SMB"], 2) == 0.4
    assert round(result["betas"]["HML"], 2) == -0.2
    assert result["alpha_t"] > 100


def test_fama_french_regression_can_run_ff3_only() -> None:
    factors = _factor_frame()
    portfolio = 0.0005 + 0.8 * factors["MKT"] + 0.2 * factors["SMB"] + 0.1 * factors["HML"]

    result = fama_french_regression(portfolio, factors, model="ff3")

    assert result["ok"] is True
    assert result["factors"] == list(FF3_FACTORS)
    assert "RMW" not in result["betas"]
    assert round(result["alpha_annual_pct"], 2) == 0.6


def test_run_factor_regressions_and_rows_flatten_results() -> None:
    factors = _factor_frame()
    portfolios = pd.DataFrame(
        {
            "base": 0.001 + factors["MKT"],
            "tilted": 0.002 + 0.7 * factors["MKT"] + 0.5 * factors["HML"],
        },
        index=factors.index,
    )

    results = run_factor_regressions(portfolios, factors, model="ff5")
    rows = factor_regression_rows(results)

    assert set(results) == {"base", "tilted"}
    assert len(rows) == 2
    assert rows[0]["portfolio"] == "base"
    assert "beta_MKT" in rows[0]


def test_cross_sectional_factor_returns_builds_korean_factor_spreads() -> None:
    index = pd.date_range("2024-01-31", periods=2, freq="ME")
    returns = pd.DataFrame(
        {
            "small_value": [0.10, 0.03],
            "small_growth": [0.08, 0.02],
            "big_value": [0.01, 0.00],
            "big_growth": [-0.01, -0.02],
        },
        index=index,
    )
    size = pd.Series(
        {"small_value": 10, "small_growth": 12, "big_value": 100, "big_growth": 120}
    )
    book_to_market = pd.Series(
        {"small_value": 2.0, "small_growth": 0.4, "big_value": 1.8, "big_growth": 0.3}
    )
    profitability = pd.Series(
        {"small_value": 0.3, "small_growth": 0.1, "big_value": 0.25, "big_growth": 0.05}
    )
    investment = pd.Series(
        {"small_value": 0.02, "small_growth": 0.20, "big_value": 0.03, "big_growth": 0.25}
    )

    factors = cross_sectional_factor_returns(
        returns,
        size=size,
        book_to_market=book_to_market,
        profitability=profitability,
        investment=investment,
        quantile=0.5,
    )

    assert list(factors.columns) == ["MKT", "SMB", "HML", "RMW", "CMA"]
    assert factors["SMB"].iloc[0] > 0
    assert factors["HML"].iloc[0] > 0
    assert factors["RMW"].iloc[0] > 0
    assert factors["CMA"].iloc[0] > 0


def test_fama_french_regression_reports_not_enough_data() -> None:
    factors = _factor_frame(3)
    portfolio = 0.001 + factors["MKT"]

    result = fama_french_regression(portfolio, factors, model="ff5")

    assert result["ok"] is False
    assert result["reason"] == "not_enough_observations"
