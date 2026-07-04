"""Repository interfaces for the gradual data-layer refactor."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd


class PriceRepository(ABC):
    """Abstract price-data source."""

    @abstractmethod
    def get_ohlcv(self, ticker: str, period: str = "1y") -> tuple[pd.DataFrame, bool]:
        """Return OHLCV data and whether it is sample/fallback data."""


class FundamentalRepository(ABC):
    """Abstract fundamental-data source."""

    @abstractmethod
    def get_info(self, ticker: str) -> tuple[dict[str, Any], bool]:
        """Return yfinance-style info and whether it is sample/fallback data."""


class UniverseRepository(ABC):
    """Abstract stock-universe source."""

    @abstractmethod
    def get_universe(self) -> dict[str, tuple[str, str]]:
        """Return Yahoo-style ticker -> (name, sector)."""
