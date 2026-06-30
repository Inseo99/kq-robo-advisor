from __future__ import annotations


def test_api_package_exports_current_helper_surface() -> None:
    import kq_tool.api as api

    expected = {
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
    }

    assert set(api.__all__) == expected
    assert all(hasattr(api, name) for name in expected)



