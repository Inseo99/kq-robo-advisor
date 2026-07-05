"""Validation reporting helpers."""

from .costs import apply_signal_costs, net_edge_return, signal_trade_cost_rate
from .factor_analysis import (
    FF3_FACTORS,
    FF5_FACTORS,
    cross_sectional_factor_returns,
    factor_regression_rows,
    fama_french_regression,
    run_factor_regressions,
)
from .deflated_sharpe import (
    annualized_sharpe,
    bootstrap_sharpe_ci,
    deflated_sharpe_ratio,
    deflated_sharpe_table,
    expected_max_sharpe,
    probabilistic_sharpe_ratio,
    sharpe_ratio,
)
from .regime_alpha_decay import (
    DEFAULT_REGIMES,
    UNKNOWN_REGIME,
    random_return_pool_by_regime,
    regime_at_date,
    regime_summary_rows,
    split_events_by_regime,
    summarize_regime_alpha_decay,
)
from .reporting import (
    ValidationRow,
    best_quality_row,
    format_signal_quality_markdown,
    load_signal_quality_summary,
    rank_quality_rows,
)
from .regime_performance import (
    best_allocation_per_regime,
    excess_vs_benchmark,
    load_inputs,
    regime_performance_matrix,
    run_full_decomposition,
    sharpe_pivot,
)
from .segments import segment_split_for_top, signal_names_for_side, ticker_segment

__all__ = [
    "DEFAULT_REGIMES",
    "FF3_FACTORS",
    "FF5_FACTORS",
    "UNKNOWN_REGIME",
    "ValidationRow",
    "annualized_sharpe",
    "apply_signal_costs",
    "best_quality_row",
    "best_allocation_per_regime",
    "bootstrap_sharpe_ci",
    "cross_sectional_factor_returns",
    "deflated_sharpe_ratio",
    "deflated_sharpe_table",
    "excess_vs_benchmark",
    "expected_max_sharpe",
    "factor_regression_rows",
    "fama_french_regression",
    "format_signal_quality_markdown",
    "load_inputs",
    "load_signal_quality_summary",
    "net_edge_return",
    "probabilistic_sharpe_ratio",
    "random_return_pool_by_regime",
    "rank_quality_rows",
    "regime_at_date",
    "regime_performance_matrix",
    "regime_summary_rows",
    "run_factor_regressions",
    "run_full_decomposition",
    "segment_split_for_top",
    "sharpe_pivot",
    "sharpe_ratio",
    "signal_trade_cost_rate",
    "signal_names_for_side",
    "split_events_by_regime",
    "summarize_regime_alpha_decay",
    "ticker_segment",
]
