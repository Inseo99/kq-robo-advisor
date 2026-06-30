from __future__ import annotations

from kq_tool.analyzer.dcf import reverse_dcf_growth


def test_reverse_dcf_growth_returns_implied_growth() -> None:
    growth = reverse_dcf_growth(price=100, eps=8, required_return=0.095)

    assert round(growth, 4) == 0.0139


def test_reverse_dcf_growth_rejects_invalid_inputs() -> None:
    assert reverse_dcf_growth(None, 8, 0.095) is None
    assert reverse_dcf_growth(100, 0, 0.095) is None


def test_reverse_dcf_growth_rejects_unrealistic_growth() -> None:
    assert reverse_dcf_growth(price=10_000, eps=1, required_return=1.0) is None
