"""Valuation helpers."""

from __future__ import annotations


def reverse_dcf_growth(price: float | None, eps: float | None, required_return: float) -> float | None:
    """Infer Gordon-growth rate implied by current price and EPS.

    Formula:
        P = EPS(1 + g) / (r - g)
        g = (rP - EPS) / (P + EPS)
    """

    if price is None or eps is None or price <= 0 or eps <= 0:
        return None
    growth = (required_return * price - eps) / (price + eps)
    if growth < -0.5 or growth > 0.5:
        return None
    return float(growth)


_reverse_dcf_growth = reverse_dcf_growth
