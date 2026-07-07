from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_launcher_batch_points_to_windows_launcher() -> None:
    content = _read("RUN_KQ_TOOL.bat")

    assert "RUN_WINDOWS.ps1" in content
    assert "ExecutionPolicy Bypass" in content
    assert 'cd /d "%~dp0"' in content


def test_regression_batch_points_to_regression_script() -> None:
    content = _read("RUN_REGRESSION_CHECKS.bat")

    assert "RUN_REGRESSION_CHECKS.ps1" in content
    assert "ExecutionPolicy Bypass" in content
    assert 'cd /d "%~dp0"' in content


def test_regression_script_includes_documented_test_count_check() -> None:
    content = _read("RUN_REGRESSION_CHECKS.ps1")

    assert "Documented test count" in content
    assert "tests\\verify_test_count_docs.py" in content


def test_team_run_guide_documents_launcher_and_core_smoke() -> None:
    content = _read("docs/TEAM_RUN_GUIDE.md")

    assert "RUN_KQ_TOOL.bat" in content
    assert "http://127.0.0.1:8888/" in content
    assert "set PYTHONPATH=%CD%\\src;%PYTHONPATH%" in content
    assert "python -m kq_tool" in content
    assert "python tests\\smoke_api.py --include-stock --include-core --timeout 180" in content


def test_readme_links_handoff_and_regression_docs() -> None:
    content = _read("README.md")

    assert "docs/TEAM_RUN_GUIDE.md" in content
    assert "docs/final-regression-checks.md" in content
    assert "docs/release-readiness.md" in content


def test_windows_launcher_prefers_supported_python_before_314() -> None:
    content = _read("RUN_WINDOWS.ps1")

    python313 = content.index("Python313\\python.exe")
    python310 = content.index("Python310\\python.exe")
    python314 = content.index("Python314\\python.exe")

    assert python313 < python310 < python314
    assert 'py"; Args = @("-3.13")' in content
    assert 'py"; Args = @("-3.10")' in content
    assert 'py"; Args = @("-3")' in content


def test_windows_launcher_keeps_server_browser_autostart_disabled() -> None:
    content = _read("RUN_WINDOWS.ps1")

    assert '$env:KQ_AUTO_OPEN_BROWSER = "0"' in content
    assert '$env:KQ_DISABLE_TABPFN = "1"' in content
    assert "Open-BrowserLater" in content


def test_windows_launcher_sets_src_pythonpath_for_package_entrypoint() -> None:
    content = _read("RUN_WINDOWS.ps1")

    assert '$SrcPath = Join-Path $Root "src"' in content
    assert "$env:PYTHONPATH = $SrcPath" in content
    assert '$env:PYTHONPATH = "$SrcPath;$env:PYTHONPATH"' in content
    assert "& $python.Command @($python.Args) -m kq_tool" in content


def test_windows_launcher_prefers_chrome_before_default_browser() -> None:
    content = _read("RUN_WINDOWS.ps1")

    chrome_paths = content.index("$paths = @(")
    chrome_lookup = content.index("$chrome = $paths")
    chrome_launch = content.index("Start-Process -FilePath $chrome")
    fallback_launch = content.index("Start-Process $url")

    assert "Google\\Chrome\\Application\\chrome.exe" in content
    assert chrome_paths < chrome_lookup < chrome_launch < fallback_launch


def test_requirements_skips_hmmlearn_on_python_314() -> None:
    content = _read("requirements.txt")

    assert 'hmmlearn>=0.3.0; python_version < "3.14"' in content

def test_server_no_longer_keeps_inline_legacy_get_route_table() -> None:
    content = _read("server.py")

    assert "from urllib.parse import urlparse, parse_qs" not in content
    assert "elif p == '/api/stock'" not in content
    assert "API routing helpers are unavailable" in content
    assert "_kq_handle_legacy_get" in content
    assert "_kq_apply_api_response" not in content
    assert "_kq_dispatch_get" not in content
    assert "handle_dispatched_get as _kq_handle_dispatched_get" not in content
    assert "response.kind" not in content

def test_server_api_services_delegates_registry_to_api_helper() -> None:
    content = _read("server.py")

    assert "from kq_tool.api.services import build_api_services as _kq_build_api_services" not in content
    assert "_kq_build_api_services" not in content
    assert "services = {" not in content
    assert "API service registry helper is unavailable" in content
    assert "_kq_build_server_api_services" in content


def test_server_endpoint_wrappers_do_not_duplicate_query_parsing() -> None:
    content = _read("server.py")

    assert "parse_stock_params as _kq_parse_stock_params" not in content
    assert "parse_strategy_backtest_params as _kq_parse_strategy_backtest_params" not in content
    assert "qs.get('t'" not in content
    assert "qs.get('s'" not in content
    assert "Stock action helper is unavailable" in content
    assert "Strategy backtest action helper is unavailable" in content

def test_server_zero_arg_endpoint_wrappers_do_not_duplicate_json_actions() -> None:
    content = _read("server.py")

    assert "self._json_action(run_screener)" not in content
    assert "self._json_action(build_regime_ai_response)" not in content
    assert "self._json_action(recommend_portfolio)" not in content
    assert "self._json_action(run_backtest)" not in content
    assert "JSON service action helper is unavailable" in content
    assert content.count("_kq_run_json_service_action(") >= 4

def test_server_json_action_does_not_keep_inline_try_except_fallback() -> None:
    content = _read("server.py")

    assert "self._json(action())" not in content
    assert "self._json(_api_error_payload(e),500)" not in content
    assert "JSON action helper is unavailable" in content
    assert "_kq_send_json_action(" in content

def test_server_json_response_does_not_keep_inline_body_header_fallback() -> None:
    content = _read("server.py")

    assert "import http.server, socketserver, json" not in content
    assert "json.dumps(_clean" not in content
    assert "json_response_body as _kq_json_response_body" not in content
    assert "JSON_CONTENT_TYPE as _KQ_JSON_CONTENT_TYPE" not in content
    assert "self.send_header('Content-Type','application/json; charset=utf-8')" not in content
    assert "JSON response helper is unavailable" in content
    assert "HTTP response body writer is unavailable" in content

def test_server_file_response_does_not_keep_inline_open_header_fallback() -> None:
    content = _read("server.py")

    assert "content_headers as _kq_content_headers" not in content
    assert "send_body_response as _kq_send_body_response" not in content
    assert "read_static_file as _kq_read_static_file" not in content
    assert "open(fp,'rb').read()" not in content
    assert "self.send_header('Content-Type',ct)" not in content
    assert "Static file helper is unavailable" in content

def test_server_json_cleaning_and_error_payload_are_helper_only() -> None:
    content = _read("server.py")

    assert "return {k:_clean(v) for k,v in obj.items()}" not in content
    assert "isinstance(obj,(np.integer,))" not in content
    assert "isinstance(obj,(np.floating,))" not in content
    assert "return {'error': str(error)}" not in content
    assert "JSON serialization helper is unavailable" in content
    assert "API error payload helper is unavailable" in content

def test_server_empty_and_header_policy_are_helper_only() -> None:
    content = _read("server.py")

    assert "self.send_header('Access-Control-Allow-Origin','*')" not in content
    assert "self.send_header('Access-Control-Allow-Methods','GET,OPTIONS')" not in content
    assert "self.send_header('Access-Control-Allow-Headers','Content-Type')" not in content
    assert "self.send_header('Cache-Control','no-store, no-cache, must-revalidate, max-age=0')" not in content
    assert "self.send_response(code); self._cors(); self.end_headers()" not in content
    assert "CORS header helper is unavailable" in content
    assert "No-cache header helper is unavailable" in content
    assert "Empty response helper is unavailable" in content

def test_server_main_uses_runtime_helpers_without_browser_or_server_fallbacks() -> None:
    content = _read("server.py")

    assert "import http.server, socketserver" not in content
    assert "webbrowser" not in content
    assert "socketserver.ThreadingMixIn" not in content
    assert "KQServer(('127.0.0.1', PORT), Handler)" not in content
    assert "threading.Timer(1.0, _open_browser).start()" not in content
    assert "srv.serve_forever()" not in content
    assert "Runtime helpers are unavailable" in content
    assert "HTTP server runtime helper is unavailable" in content
    assert "_kq_create_http_server(KQServer, Handler" in content
    assert "_kq_run_server_with_browser_policy(srv, os.environ, url)" in content




