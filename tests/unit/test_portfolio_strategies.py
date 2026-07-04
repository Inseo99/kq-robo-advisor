from kq_tool.config import (
    EQUITY_RISK_PREMIUM,
    ETFS,
    HORIZONS,
    RISK_BASED_KEYS,
    RISK_FREE_RATE,
    ROBO_BUY_THRESHOLD,
    ROBO_SELL_THRESHOLD,
    ROBO_SIGNAL_LABELS,
    ROBO_SIGNAL_WEIGHTS,
    SCREENER_LIMIT,
    SIGNAL_DIRECTION,
    STRATEGIES,
)
from kq_tool.portfolio.allocation import (
    evaluate_asset_allocation_strategies,
    gtaa_weights,
    inverse_volatility_weights,
    risk_based_strategy_weights,
)
from kq_tool.portfolio.recommender import META_COMPONENTS, REGIME_TARGETS


def test_asset_allocation_strategy_count_is_ten():
    assert len(STRATEGIES) == 10


def test_static_6040_strategy_weights_sum_to_one():
    weights = STRATEGIES["정적 60/40"]

    assert weights == {
        "069500.KS": 0.60,
        "148070.KS": 0.40,
    }
    assert sum(weights.values()) == 1.0


def test_server_reuses_config_strategy_definitions():
    import server

    assert server.ETFs == ETFS
    assert server.RISK_BASED_KEYS == RISK_BASED_KEYS
    assert set(server.STRATEGIES) == set(STRATEGIES)
    assert server.STRATEGIES["정적 60/40"] == STRATEGIES["정적 60/40"]
    assert server._kq_evaluate_asset_allocation_strategies is evaluate_asset_allocation_strategies
    assert server._kq_inverse_volatility_weights is inverse_volatility_weights
    assert server._kq_gtaa_weights is gtaa_weights
    assert server._kq_risk_based_strategy_weights is risk_based_strategy_weights


def test_server_reuses_config_market_assumptions_and_signals():
    import server

    assert server.RF == RISK_FREE_RATE
    assert server.ERP == EQUITY_RISK_PREMIUM
    assert server.HORIZONS == HORIZONS
    assert server.SCREENER_LIMIT == SCREENER_LIMIT
    assert server.ROBO_BUY_THRESHOLD == ROBO_BUY_THRESHOLD
    assert server.ROBO_SELL_THRESHOLD == ROBO_SELL_THRESHOLD
    assert server.SIGNAL_DIRECTION == SIGNAL_DIRECTION
    assert server.ROBO_SIGNAL_WEIGHTS == ROBO_SIGNAL_WEIGHTS
    assert server.ROBO_SIGNAL_LABELS == ROBO_SIGNAL_LABELS


def test_server_reuses_recommendation_settings():
    import server

    assert server.META_COMPONENTS == META_COMPONENTS
    assert server.REGIME_TARGETS == REGIME_TARGETS
