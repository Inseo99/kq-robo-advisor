"""Service registry helpers for API route dispatch."""

from __future__ import annotations

from collections.abc import Callable, Mapping

REQUIRED_SERVICE_KEYS = (
    "health",
    "macro",
    "stock",
    "screen",
    "backtest",
    "stratbt",
    "regime_ai",
    "recommend_portfolio",
    "market_report",
)

OPTIONAL_SERVICE_KEYS = (
    "portfolio_orders",
    "return_heatmap",
)


def missing_service_keys(services: Mapping[str, object]) -> tuple[str, ...]:
    """Return required route service names that are absent or not callable."""

    return tuple(
        key for key in REQUIRED_SERVICE_KEYS if key not in services or not callable(services[key])
    )


def build_api_services(**services: Callable[..., object]) -> dict[str, Callable[..., object]]:
    """Validate and return the service mapping expected by the dispatcher."""

    missing = missing_service_keys(services)
    if missing:
        raise KeyError(f"Missing API services: {', '.join(missing)}")
    registry = {key: services[key] for key in REQUIRED_SERVICE_KEYS}
    for key in OPTIONAL_SERVICE_KEYS:
        service = services.get(key)
        if callable(service):
            registry[key] = service
    return registry


def build_server_api_services(
    *,
    health: Callable[[], object],
    macro_payload: object,
    stock: Callable[..., object],
    screen: Callable[[], object],
    backtest: Callable[[], object],
    stratbt: Callable[..., object],
    regime_ai: Callable[[], object],
    recommend_portfolio: Callable[[], object],
    market_report: Callable[[], object],
    portfolio_orders: Callable[..., object] | None = None,
    return_heatmap: Callable[..., object] | None = None,
) -> dict[str, Callable[..., object]]:
    """Build the service registry used by the local server handler."""

    return build_api_services(
        health=health,
        macro=lambda: macro_payload,
        stock=stock,
        screen=screen,
        backtest=backtest,
        stratbt=stratbt,
        regime_ai=regime_ai,
        recommend_portfolio=recommend_portfolio,
        market_report=market_report,
        portfolio_orders=portfolio_orders,
        return_heatmap=return_heatmap,
    )
