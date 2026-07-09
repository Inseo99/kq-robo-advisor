"""Backtest security selection helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from kq_tool.backtest.strategy_meta import (
    QUANT,
    QUANT_ROBO_FILTER,
    QUANT_S2,
    QUANT_S2_ROBO_FILTER,
    ROBO,
    normalize_strategy_key,
)


def latest_at_or_before(frame: pd.DataFrame, date: object) -> pd.Series:
    """Return the latest row whose index is at or before ``date``."""

    idx = frame.index[frame.index <= date]
    if len(idx) == 0:
        return pd.Series(np.nan, index=frame.columns)
    return frame.loc[idx[-1]]


def select_momentum(hist: pd.DataFrame, top_n: int) -> list[str]:
    """Select tickers by recent 3-month price momentum."""

    if len(hist) < 60:
        return []
    momentum = hist.iloc[-1] / hist.iloc[-60] - 1
    momentum = momentum.dropna()
    return momentum.nlargest(top_n).index.tolist()


def select_s2_momentum(hist: pd.DataFrame, top_n: int) -> list[str]:
    """Select tickers by S2 12-1 month momentum."""

    if len(hist) < 252:
        return []
    momentum = hist.iloc[-21] / hist.iloc[-252] - 1
    momentum = momentum.dropna()
    return momentum.nlargest(top_n).index.tolist()


def select_momentum_ranked(hist: pd.DataFrame, limit: int) -> list[str]:
    """Return recent-momentum candidates ordered from strongest to weakest."""

    return select_momentum(hist, limit)


def select_s2_momentum_ranked(hist: pd.DataFrame, limit: int) -> list[str]:
    """Return S2 12-1 momentum candidates ordered from strongest to weakest."""

    return select_s2_momentum(hist, limit)


def build_robo_precomputed_indicators(
    price_df: pd.DataFrame,
    *,
    min_history: int = 60,
) -> dict[str, pd.DataFrame]:
    """Precompute RSI/MACD/MA tables for the legacy robo backtest strategy."""

    rsi_dict: dict[str, pd.Series] = {}
    macd_bull_dict: dict[str, pd.Series] = {}
    ma20_dict: dict[str, pd.Series] = {}
    ma60_dict: dict[str, pd.Series] = {}

    for col in price_df.columns:
        series = price_df[col].dropna()
        if len(series) < min_history:
            continue

        delta = series.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(14, min_periods=14).mean()
        avg_loss = loss.rolling(14, min_periods=14).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi_dict[col] = 100 - (100 / (1 + rs))

        ema12 = series.ewm(span=12, adjust=False).mean()
        ema26 = series.ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        macd_bull_dict[col] = (macd_line > signal_line).astype(int)

        ma20_dict[col] = series.rolling(20).mean()
        ma60_dict[col] = series.rolling(60).mean()

    return {
        "rsi": pd.DataFrame(rsi_dict),
        "macd_bull": pd.DataFrame(macd_bull_dict),
        "ma20": pd.DataFrame(ma20_dict),
        "ma60": pd.DataFrame(ma60_dict),
    }


def robo_scores_from_indicators(
    hist: pd.DataFrame,
    rsi_last: pd.Series,
    macd_bull: pd.Series,
    ma20: pd.Series,
    ma60: pd.Series,
    cur: pd.Series,
) -> pd.Series:
    """Compute vectorized robo scores aligned to ``hist.columns``."""

    cols = hist.columns
    rsi_last = pd.to_numeric(pd.Series(rsi_last).reindex(cols), errors="coerce")
    macd_bull = pd.Series(macd_bull).reindex(cols).fillna(False).astype(bool)
    ma20 = pd.to_numeric(pd.Series(ma20).reindex(cols), errors="coerce")
    ma60 = pd.to_numeric(pd.Series(ma60).reindex(cols), errors="coerce")
    cur = pd.to_numeric(pd.Series(cur).reindex(cols), errors="coerce")

    rsi_values = rsi_last.to_numpy(dtype=float)
    macd_values = macd_bull.to_numpy(dtype=bool)
    ma20_values = ma20.to_numpy(dtype=float)
    ma60_values = ma60.to_numpy(dtype=float)
    cur_values = cur.to_numpy(dtype=float)

    rsi_score = np.where(rsi_values < 30, 20, np.where(rsi_values > 70, -20, 0))
    macd_score = np.where(macd_values, 25, -25)
    ma20_score = np.where(cur_values > ma20_values, 15, -15)
    ma60_score = np.where(cur_values > ma60_values, 20, -20)

    total_score = pd.Series(rsi_score + macd_score + ma20_score + ma60_score, index=cols)
    valid_mask = rsi_last.notna() & ma20.notna() & ma60.notna() & cur.notna()
    return total_score[valid_mask]


def select_robo(
    hist: pd.DataFrame,
    top_n: int,
    precomputed: dict[str, pd.DataFrame] | None = None,
    cur_date: object | None = None,
) -> list[str]:
    """Select tickers by the legacy robo scoring rule."""

    if len(hist) < 60:
        return []

    if precomputed is not None and cur_date is not None:
        rsi_last = latest_at_or_before(precomputed["rsi"], cur_date)
        macd_bull = latest_at_or_before(precomputed["macd_bull"], cur_date) == 1
        ma20 = latest_at_or_before(precomputed["ma20"], cur_date)
        ma60 = latest_at_or_before(precomputed["ma60"], cur_date)
        cur = hist.iloc[-1]
    else:
        delta = hist.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(14, min_periods=14).mean()
        avg_loss = loss.rolling(14, min_periods=14).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        rsi_last = rsi.iloc[-1]
        ema12 = hist.ewm(span=12, adjust=False).mean()
        ema26 = hist.ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        macd_bull = macd_line.iloc[-1] > signal_line.iloc[-1]
        ma20 = hist.rolling(20).mean().iloc[-1]
        ma60 = hist.rolling(60).mean().iloc[-1]
        cur = hist.iloc[-1]

    total_score = robo_scores_from_indicators(hist, rsi_last, macd_bull, ma20, ma60, cur)
    positive = total_score[total_score > 0]
    if len(positive) >= top_n:
        return positive.nlargest(top_n).index.tolist()
    return total_score.nlargest(top_n).index.tolist()


def robo_scores_at_date(
    hist: pd.DataFrame,
    precomputed: dict[str, pd.DataFrame] | None = None,
    cur_date: object | None = None,
) -> pd.Series:
    """Return robo scores for the current backtest date aligned to ``hist``."""

    if len(hist) < 60:
        return pd.Series(dtype=float)
    if precomputed is not None and cur_date is not None:
        rsi_last = latest_at_or_before(precomputed["rsi"], cur_date)
        macd_bull = latest_at_or_before(precomputed["macd_bull"], cur_date) == 1
        ma20 = latest_at_or_before(precomputed["ma20"], cur_date)
        ma60 = latest_at_or_before(precomputed["ma60"], cur_date)
        cur = hist.iloc[-1]
        return robo_scores_from_indicators(hist, rsi_last, macd_bull, ma20, ma60, cur)

    delta = hist.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(14, min_periods=14).mean()
    avg_loss = loss.rolling(14, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    ema12 = hist.ewm(span=12, adjust=False).mean()
    ema26 = hist.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    ma20 = hist.rolling(20).mean().iloc[-1]
    ma60 = hist.rolling(60).mean().iloc[-1]
    return robo_scores_from_indicators(
        hist,
        rsi.iloc[-1],
        macd_line.iloc[-1] > signal_line.iloc[-1],
        ma20,
        ma60,
        hist.iloc[-1],
    )


def select_robo_filtered_momentum(
    hist: pd.DataFrame,
    *,
    base_strategy: str,
    top_n: int,
    precomputed: dict[str, pd.DataFrame] | None = None,
    cur_date: object | None = None,
    pool_mult: float = 2.0,
) -> list[str]:
    """Select momentum candidates after excluding bearish robo-score names.

    The robo score is used only as a filter. Candidates are still ranked by the
    underlying quant strategy, and replacements come from at most ``pool_mult``
    times the target count to keep the strategy recognizably momentum-based.
    """

    return list(
        robo_filter_decision(
            hist,
            base_strategy=base_strategy,
            top_n=top_n,
            precomputed=precomputed,
            cur_date=cur_date,
            pool_mult=pool_mult,
        )["selected"]
    )


def robo_filter_decision(
    hist: pd.DataFrame,
    *,
    base_strategy: str,
    top_n: int,
    precomputed: dict[str, pd.DataFrame] | None = None,
    cur_date: object | None = None,
    pool_mult: float = 2.0,
) -> dict:
    """Return the robo-filter selection plus exclusion/replacement audit fields."""

    pool_size = max(top_n, int(round(top_n * pool_mult)))
    if base_strategy == QUANT_S2.key:
        ranked = select_s2_momentum_ranked(hist, pool_size)
    else:
        ranked = select_momentum_ranked(hist, pool_size)
    if not ranked:
        return {
            "base_strategy": base_strategy,
            "ranked": [],
            "raw_selected": [],
            "selected": [],
            "excluded": [],
            "replacements": [],
            "scores": {},
        }

    score_hist = hist[[ticker for ticker in ranked if ticker in hist.columns]]
    scores = robo_scores_at_date(score_hist, precomputed, cur_date)
    raw_selected = ranked[:top_n]
    picked: list[str] = []
    for ticker in ranked:
        score = scores.get(ticker)
        if pd.notna(score) and float(score) < 0:
            continue
        picked.append(ticker)
        if len(picked) >= top_n:
            break
    excluded = [ticker for ticker in raw_selected if ticker not in picked]
    replacements = [ticker for ticker in picked if ticker not in raw_selected]
    return {
        "base_strategy": base_strategy,
        "ranked": ranked,
        "raw_selected": raw_selected,
        "selected": picked,
        "excluded": excluded,
        "replacements": replacements,
        "scores": {
            ticker: round(float(score), 4)
            for ticker, score in scores.reindex(ranked).dropna().items()
        },
    }


def select_for_backtest(
    hist: pd.DataFrame,
    strategy: str,
    top_n: int,
    precomputed: dict[str, pd.DataFrame] | None = None,
    cur_date: object | None = None,
) -> list[str]:
    """Select tickers for a strategy backtest."""

    strategy_key = normalize_strategy_key(strategy)
    if strategy_key == QUANT.key:
        return select_momentum(hist, top_n)
    if strategy_key == QUANT_S2.key:
        return select_s2_momentum(hist, top_n)
    if strategy_key == QUANT_ROBO_FILTER.key:
        return select_robo_filtered_momentum(
            hist,
            base_strategy=QUANT.key,
            top_n=top_n,
            precomputed=precomputed,
            cur_date=cur_date,
        )
    if strategy_key == QUANT_S2_ROBO_FILTER.key:
        return select_robo_filtered_momentum(
            hist,
            base_strategy=QUANT_S2.key,
            top_n=top_n,
            precomputed=precomputed,
            cur_date=cur_date,
        )
    if strategy_key == ROBO.key:
        return select_robo(hist, top_n, precomputed, cur_date)
    return list(hist.columns[:top_n])


_select_for_bt = select_for_backtest
_build_robo_precomputed_indicators = build_robo_precomputed_indicators
