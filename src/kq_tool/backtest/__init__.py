"""Backtest helpers."""

from .costs import (
    apply_transaction_cost,
    bps_to_rate,
    equal_weights,
    portfolio_turnover,
    transaction_cost_rate,
)
from .comparison import build_quant_comparison_response
from .engine import equal_weight_period_return, normalize_benchmark_to_equity, underwater_curve
from .metrics import perf_metrics
from .orchestrator import run_rebalanced_strategy_backtest
from .preparation import (
    evaluation_start_for_period,
    filter_strategy_benchmark_period,
    filter_strategy_price_period,
    prepare_strategy_price_frame,
    use_fixed_start_for_period,
)
from .selector import (
    build_robo_precomputed_indicators,
    select_for_backtest,
    select_momentum,
    select_robo,
    select_s2_momentum,
)
from .strategy_meta import (
    KOSPI_BENCHMARK,
    QUANT,
    QUANT_COMPARE,
    QUANT_S2,
    ROBO,
    StrategyDescriptor,
    normalize_strategy_key,
    strategy_descriptor,
)

__all__ = [
    "equal_weight_period_return",
    "apply_transaction_cost",
    "bps_to_rate",
    "build_robo_precomputed_indicators",
    "build_quant_comparison_response",
    "equal_weights",
    "normalize_benchmark_to_equity",
    "normalize_strategy_key",
    "perf_metrics",
    "portfolio_turnover",
    "evaluation_start_for_period",
    "filter_strategy_benchmark_period",
    "filter_strategy_price_period",
    "prepare_strategy_price_frame",
    "run_rebalanced_strategy_backtest",
    "select_for_backtest",
    "select_momentum",
    "select_robo",
    "select_s2_momentum",
    "strategy_descriptor",
    "KOSPI_BENCHMARK",
    "QUANT",
    "QUANT_COMPARE",
    "QUANT_S2",
    "ROBO",
    "StrategyDescriptor",
    "transaction_cost_rate",
    "underwater_curve",
    "use_fixed_start_for_period",
]

