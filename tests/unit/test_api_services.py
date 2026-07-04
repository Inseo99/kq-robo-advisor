from __future__ import annotations

import pytest

from kq_tool.api.services import (
    REQUIRED_SERVICE_KEYS,
    build_api_services,
    build_server_api_services,
    missing_service_keys,
)


def _callable() -> dict[str, bool]:
    return {"ok": True}


def test_build_api_services_returns_required_services_in_stable_order() -> None:
    services = {key: _callable for key in REQUIRED_SERVICE_KEYS}

    result = build_api_services(**services)

    assert tuple(result) == REQUIRED_SERVICE_KEYS
    assert all(callable(value) for value in result.values())


def test_missing_service_keys_reports_absent_and_noncallable_services() -> None:
    services = {key: _callable for key in REQUIRED_SERVICE_KEYS}
    services.pop("stock")
    services["screen"] = {"not": "callable"}

    assert missing_service_keys(services) == ("stock", "screen")


def test_build_api_services_raises_when_contract_is_incomplete() -> None:
    services = {key: _callable for key in REQUIRED_SERVICE_KEYS}
    services.pop("regime_ai")

    with pytest.raises(KeyError, match="regime_ai"):
        build_api_services(**services)


def test_build_server_api_services_wraps_macro_payload_and_keeps_order() -> None:
    macro = {"kospi": 1}
    services = build_server_api_services(
        health=_callable,
        macro_payload=macro,
        stock=lambda ticker, period: {"ticker": ticker, "period": period},
        screen=_callable,
        backtest=_callable,
        stratbt=lambda *args: {"args": args},
        regime_ai=_callable,
        recommend_portfolio=_callable,
        market_report=_callable,
    )

    assert tuple(services) == REQUIRED_SERVICE_KEYS
    assert services["macro"]() is macro
    assert services["stock"]("005930.KS", "1y") == {"ticker": "005930.KS", "period": "1y"}


def test_server_reuses_server_api_services_helper() -> None:
    import server
    from kq_tool.api.services import build_server_api_services

    assert server._kq_build_server_api_services is build_server_api_services
