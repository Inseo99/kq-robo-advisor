"""Point-in-time macro data loader for regime classification.

All macro series used for regime labels should be accessed through
``get_observable_panel()``. Raw series are indexed by the reference month, while
observable series are indexed by the decision month. The invariant is:

    observable[t] == raw[t - lag]

where lag is the indicator-specific publication delay plus revision buffer from
``INDICATOR_REGISTRY``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

__all__ = [
    "IndicatorSpec",
    "INDICATOR_REGISTRY",
    "clear_raw_sources",
    "get_observable_panel",
    "register_raw_source",
    "to_observable",
]


@dataclass(frozen=True)
class IndicatorSpec:
    """Publication-lag specification for one macro/market indicator."""

    key: str
    name_kr: str
    source: str
    freq: str
    publication_lag_m: int
    revision_buffer_m: int = 0
    notes: str = ""

    @property
    def total_lag_m(self) -> int:
        return int(self.publication_lag_m + self.revision_buffer_m)


INDICATOR_REGISTRY: dict[str, IndicatorSpec] = {
    spec.key: spec
    for spec in [
        IndicatorSpec("cpi_yoy", "소비자물가 상승률", "KOSIS/ECOS", "M", 1, 0, "익월 초 발표"),
        IndicatorSpec("exports_yoy", "수출 증가율", "산업통상자원부", "M", 1, 0, "익월 1일 발표"),
        IndicatorSpec(
            "leading_index_cycle",
            "경기선행지수 순환변동치",
            "통계청 산업활동동향",
            "M",
            1,
            1,
            "익월 말 발표 + 소급 수정. vintage 확보 전까지 revision bias 주의",
        ),
        IndicatorSpec("industrial_production", "광공업생산", "통계청 산업활동동향", "M", 1, 1, "익월 말 발표 + 수정"),
        IndicatorSpec("unemployment_rate", "실업률", "통계청 고용동향", "M", 1, 0, "익월 중순 발표"),
        IndicatorSpec("gdp_qoq", "GDP 성장률(속보)", "한국은행", "Q", 1, 1, "분기 종료 후 약 1개월 + 속보/잠정/확정 수정"),
        IndicatorSpec("base_rate", "한국은행 기준금리", "한국은행", "D", 0, 0, "실시간"),
        IndicatorSpec("treasury_3y", "국고채 3년 금리", "시장", "D", 0, 0, "실시간"),
        IndicatorSpec("yield_spread_10y_3y", "장단기 금리차(10y-3y)", "시장", "D", 0, 0, "실시간"),
        IndicatorSpec("credit_spread", "신용스프레드(AA- 회사채-국고채)", "시장", "D", 0, 0, "실시간"),
        IndicatorSpec("usdkrw", "원/달러 환율", "시장", "D", 0, 0, "실시간"),
        IndicatorSpec("kospi", "KOSPI 지수", "시장", "D", 0, 0, "실시간"),
    ]
}


_RAW_SOURCES: dict[str, Callable[[], pd.Series]] = {}
_DEFAULT_DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "macro"


def register_raw_source(key: str, loader: Callable[[], pd.Series]) -> None:
    """Register a raw source loader for an indicator.

    This keeps the PiT logic independent from whether the data came from CSV,
    ECOS cache, Excel, or tests.
    """
    if key not in INDICATOR_REGISTRY:
        raise KeyError(
            f"'{key}' is not registered. Add it to docs/macro_publication_lags.md "
            "and INDICATOR_REGISTRY before using it."
        )
    _RAW_SOURCES[key] = loader


def clear_raw_sources() -> None:
    """Clear registered raw loaders. Mostly useful for validation/tests."""
    _RAW_SOURCES.clear()


def _pick_columns(df: pd.DataFrame) -> tuple[str, str]:
    if {"date", "value"} <= set(df.columns):
        return "date", "value"

    date_col = None
    for col in df.columns:
        parsed = pd.to_datetime(df[col], errors="coerce")
        if parsed.notna().sum() >= max(1, int(len(df) * 0.5)):
            date_col = col
            break
    if date_col is None:
        date_col = df.columns[0]

    value_candidates = [col for col in df.columns if col != date_col]
    numeric_scores = {
        col: pd.to_numeric(df[col], errors="coerce").notna().sum()
        for col in value_candidates
    }
    if not numeric_scores:
        raise ValueError("CSV must contain a date column and at least one numeric value column.")
    value_col = max(numeric_scores, key=numeric_scores.get)
    return str(date_col), str(value_col)


def _default_csv_loader(key: str) -> pd.Series:
    path = _DEFAULT_DATA_DIR / f"{key}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Raw macro file not found: {path}\n"
            f"Prepare data/macro/{key}.csv (date,value) or call register_raw_source('{key}', loader)."
        )

    df = pd.read_csv(path)
    date_col, value_col = _pick_columns(df)
    series = pd.Series(
        pd.to_numeric(df[value_col], errors="coerce").values,
        index=pd.to_datetime(df[date_col], errors="coerce"),
        name=key,
    ).dropna()
    series = series[~series.index.isna()]
    return series


def _load_raw_series(key: str) -> pd.Series:
    """Internal raw loader. Regime code should not call this directly."""
    if key not in INDICATOR_REGISTRY:
        raise KeyError(f"Unknown indicator '{key}'. Register publication lag first.")

    loader = _RAW_SOURCES.get(key)
    raw = loader() if loader is not None else _default_csv_loader(key)
    raw = pd.to_numeric(raw, errors="coerce").dropna()
    raw.index = pd.to_datetime(raw.index)

    spec = INDICATOR_REGISTRY[key]
    if spec.freq == "D":
        monthly = raw.resample("ME").last()
    elif spec.freq in ("M", "Q"):
        monthly = raw.copy()
        monthly.index = monthly.index.to_period("M").to_timestamp("M")
        monthly = monthly.resample("ME").last()
    else:
        raise ValueError(f"Unsupported frequency for {key}: {spec.freq}")

    monthly.name = key
    return monthly.dropna()


def _load_raw_frame(keys: list[str]) -> pd.DataFrame:
    """Internal raw panel loader. Public code should use get_observable_panel()."""
    return pd.concat([_load_raw_series(key) for key in keys], axis=1)


def to_observable(raw: pd.Series, key: str) -> pd.Series:
    """Convert a reference-month raw series into a decision-month observable series."""
    if key not in INDICATOR_REGISTRY:
        raise KeyError(f"Unknown indicator '{key}'. Register publication lag first.")

    spec = INDICATOR_REGISTRY[key]
    monthly = raw.copy()
    monthly.index = pd.to_datetime(monthly.index)
    monthly = monthly.sort_index()
    if monthly.index.freq is None:
        monthly = monthly.asfreq("ME")

    obs = monthly.shift(spec.total_lag_m)
    if spec.freq == "Q":
        obs = obs.ffill()

    obs.name = key
    return obs


def get_observable_panel(
    keys: list[str] | None = None,
    asof: pd.Timestamp | str | None = None,
) -> pd.DataFrame:
    """Return a PiT-safe monthly observable macro panel."""
    selected = list(INDICATOR_REGISTRY) if keys is None else list(keys)
    unknown = [key for key in selected if key not in INDICATOR_REGISTRY]
    if unknown:
        raise KeyError(
            f"Unknown indicators {unknown}. Register them in docs/macro_publication_lags.md "
            "and INDICATOR_REGISTRY first."
        )

    panel = pd.concat([to_observable(_load_raw_series(key), key) for key in selected], axis=1)

    if asof is not None:
        asof_ts = pd.Timestamp(asof)
        cutoff = asof_ts.to_period("M").to_timestamp("M")
        if asof_ts < cutoff:
            cutoff = cutoff - pd.offsets.MonthEnd(1)
        panel = panel.loc[:cutoff]

    return panel
