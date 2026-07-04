from __future__ import annotations

import numpy as np

from kq_tool.portfolio.risk_based import (
    diversification_ratio,
    erc_weights,
    gmv_weights,
    mdp_weights,
    risk_contributions,
)


def _cov() -> np.ndarray:
    return np.array(
        [
            [0.04, 0.01, 0.00],
            [0.01, 0.09, 0.01],
            [0.00, 0.01, 0.16],
        ],
        dtype=float,
    )


def test_risk_contributions_sum_to_portfolio_volatility() -> None:
    weights = np.array([0.4, 0.3, 0.3])

    contributions, sigma = risk_contributions(weights, _cov())

    assert np.isclose(contributions.sum(), sigma)


def test_gmv_weights_sum_to_one() -> None:
    weights = gmv_weights(_cov())

    assert np.isclose(weights.sum(), 1.0)
    assert (weights >= 0).all()


def test_mdp_weights_sum_to_one() -> None:
    cov = _cov()
    vols = np.sqrt(np.diag(cov))

    weights = mdp_weights(cov, vols)

    assert np.isclose(weights.sum(), 1.0)
    assert (weights >= 0).all()


def test_erc_weights_sum_to_one() -> None:
    weights = erc_weights(_cov())

    assert np.isclose(weights.sum(), 1.0)
    assert (weights >= 0).all()


def test_diversification_ratio_is_positive() -> None:
    cov = _cov()
    weights = np.array([1 / 3, 1 / 3, 1 / 3])
    vols = np.sqrt(np.diag(cov))

    assert diversification_ratio(weights, cov, vols) > 0
