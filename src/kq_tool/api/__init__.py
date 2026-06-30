"""API route helpers."""

from .dispatcher import (
    ApiResponse,
    apply_api_response,
    dispatch_get,
    handle_dispatched_get,
    handle_dispatched_get_safely,
    handle_legacy_get,
)
from .health import (
    build_data_status,
    build_health_payload,
    build_module_status,
    build_regime_status,
    build_server_health_payload,
)
from .http_response import (
    JSON_CONTENT_TYPE,
    content_headers,
    cors_headers,
    error_payload,
    json_response_body,
    no_cache_headers,
    send_body_response,
    send_empty_response,
    send_headers,
    send_json_action,
    send_json_response,
)
from .params import (
    StockParams,
    StrategyBacktestParams,
    first_query_value,
    parse_stock_params,
    parse_strategy_backtest_params,
)
from .routes import build_get_response
from .runtime import (
    ThreadingReusableHTTPServer,
    browser_autostart_disabled_message,
    browser_candidates,
    create_http_server,
    open_preferred_browser,
    run_server_with_browser_policy,
    schedule_browser_open,
    serve_until_interrupted,
    server_ready_messages,
    server_url,
    should_auto_open_browser,
    startup_intro_messages,
    yfinance_status_messages,
)
from .services import (
    REQUIRED_SERVICE_KEYS,
    build_api_services,
    build_server_api_services,
    missing_service_keys,
)
from .serialization import clean_json_value
from .static_files import StaticFilePayload, read_static_file

__all__ = [
    "ApiResponse",
    "JSON_CONTENT_TYPE",
    "REQUIRED_SERVICE_KEYS",
    "StaticFilePayload",
    "StockParams",
    "StrategyBacktestParams",
    "ThreadingReusableHTTPServer",
    "apply_api_response",
    "browser_autostart_disabled_message",
    "browser_candidates",
    "build_api_services",
    "build_data_status",
    "build_get_response",
    "build_health_payload",
    "build_module_status",
    "build_regime_status",
    "build_server_api_services",
    "build_server_health_payload",
    "clean_json_value",
    "content_headers",
    "cors_headers",
    "create_http_server",
    "dispatch_get",
    "error_payload",
    "first_query_value",
    "handle_dispatched_get",
    "handle_dispatched_get_safely",
    "handle_legacy_get",
    "json_response_body",
    "missing_service_keys",
    "no_cache_headers",
    "open_preferred_browser",
    "parse_stock_params",
    "parse_strategy_backtest_params",
    "read_static_file",
    "run_server_with_browser_policy",
    "schedule_browser_open",
    "send_body_response",
    "send_empty_response",
    "send_headers",
    "send_json_action",
    "send_json_response",
    "serve_until_interrupted",
    "server_ready_messages",
    "server_url",
    "should_auto_open_browser",
    "startup_intro_messages",
    "yfinance_status_messages",
]



