"""Chart response builders for stock analysis."""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from kq_tool.analyzer.indicators import close_series


def _numeric_series(values: object, index: pd.Index) -> pd.Series:
    if values is None:
        return pd.Series(index=index, dtype="float64")
    series = pd.Series(values).squeeze()
    if isinstance(series, pd.DataFrame):
        series = series.iloc[:, 0]
    if not series.index.equals(index):
        series = series.reindex(index)
    return pd.to_numeric(series, errors="coerce")


def _rounded(values: object, index: pd.Index) -> list[float | None]:
    series = _numeric_series(values, index)
    return [round(float(v), 2) if pd.notna(v) else None for v in series.tolist()]


def _integer(values: object, index: pd.Index) -> list[int]:
    series = _numeric_series(values, index)
    return [int(v) if pd.notna(v) else 0 for v in series.tolist()]


def _frame_column(frame: pd.DataFrame, name: str, index: pd.Index) -> pd.Series:
    if name not in frame.columns:
        return pd.Series(index=index, dtype="float64")
    return _numeric_series(frame[name].squeeze(), index)


def build_stock_chart(
    frame: pd.DataFrame,
    close: pd.Series,
    indicators: Mapping[str, object] | None = None,
    period: str = "1y",
) -> dict:
    """Build the complete chart payload without truncating server-side rows."""

    close = pd.to_numeric(close_series(close), errors="coerce")
    index = close.index
    indicators = indicators or {}
    hist = indicators.get("hist")
    if hist is None and "macd" in indicators and "signal" in indicators:
        hist = _numeric_series(indicators["macd"], index) - _numeric_series(
            indicators["signal"], index
        )

    return {
        "period": period,
        "count": int(len(close)),
        "dates": [d.strftime("%Y-%m-%d") for d in index],
        "open": _rounded(_frame_column(frame, "Open", index), index),
        "high": _rounded(_frame_column(frame, "High", index), index),
        "low": _rounded(_frame_column(frame, "Low", index), index),
        "close": _rounded(close, index),
        "ma20": _rounded(indicators.get("ma20"), index),
        "ma60": _rounded(indicators.get("ma60"), index),
        "ma200": _rounded(indicators.get("ma200"), index),
        "bb_up": _rounded(indicators.get("bb_up"), index),
        "bb_mid": _rounded(indicators.get("bb_mid"), index),
        "bb_lo": _rounded(indicators.get("bb_lo"), index),
        "vol": _integer(_frame_column(frame, "Volume", index), index),
        "rsi": _rounded(indicators.get("rsi"), index),
        "macd": _rounded(indicators.get("macd"), index),
        "signal": _rounded(indicators.get("signal"), index),
        "hist": _rounded(hist, index),
    }


_build_stock_chart = build_stock_chart
