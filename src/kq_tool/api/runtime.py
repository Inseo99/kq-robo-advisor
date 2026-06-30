"""Runtime helpers for the local HTTP app."""

from __future__ import annotations

import http.server
import socketserver
from collections.abc import Callable, Mapping
from typing import Any

CHROME_BROWSER_NAMES = ("chrome", "google-chrome", "chromium", "chromium-browser")
BROWSER_AUTOSTART_DISABLED_MESSAGE = "브라우저 자동 열기 꺼짐: Edge가 뜨지 않도록 기본값을 변경했습니다."


class ThreadingReusableHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    """Threaded local HTTP server with dev-friendly address reuse."""

    daemon_threads = True
    allow_reuse_address = True


def startup_intro_messages() -> tuple[str, ...]:
    """Return banner lines printed before slow startup checks."""

    return (
        "\n=== KQ Quant Tool ===",
        "Yahoo Finance 연결 확인 중...",
    )


def yfinance_status_messages(connected: bool) -> tuple[str, ...]:
    """Return startup lines describing Yahoo Finance connectivity."""

    if connected:
        return ("OK 연결 성공 - 실시간 데이터로 동작합니다",)
    return (
        "WARN 연결 실패 - 샘플(시뮬레이션) 데이터로 동작합니다",
        "   네트워크/방화벽이 finance.yahoo.com, fc.yahoo.com 접속을 막고 있는지 확인해 주세요",
        "   (pip install --upgrade yfinance curl_cffi 로도 해결되는 경우가 많습니다)",
    )


def server_ready_messages(url: str) -> tuple[str, ...]:
    """Return startup lines shown once the local server URL is known."""

    return (
        url,
        "Chrome에서 위 주소를 열어 사용하세요.",
        "Ctrl+C 로 종료\n",
    )


def browser_autostart_disabled_message() -> str:
    """Return the user-facing message for the default no-auto-browser policy."""

    return BROWSER_AUTOSTART_DISABLED_MESSAGE


def serve_until_interrupted(
    server: Any,
    *,
    output: Callable[[str], object] | None = print,
    interrupt_message: str = "\n종료",
) -> dict[str, bool]:
    """Run ``server.serve_forever()`` and handle Ctrl+C consistently."""

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        if output is not None:
            output(interrupt_message)
        return {"interrupted": True}
    return {"interrupted": False}


def server_url(host: str, port: int | str) -> str:
    """Build the local server URL shown to users and browser launchers."""

    return f"http://{host}:{int(port)}"


def create_http_server(
    server_cls: Callable[[tuple[str, int], Any], Any],
    handler_cls: Any,
    *,
    host: str,
    port: int | str,
) -> Any:
    """Create the HTTP server instance used by the local app."""

    return server_cls((host, int(port)), handler_cls)


def should_auto_open_browser(env: Mapping[str, str]) -> bool:
    """Return whether the server should open a browser on startup."""

    return str(env.get("KQ_AUTO_OPEN_BROWSER", "0")).strip().lower() in {"1", "true", "yes"}


def browser_candidates(preferred_browser: str | None) -> tuple[str, ...]:
    """Return named webbrowser candidates for an optional user preference."""

    preferred = (preferred_browser or "").strip().lower()
    if preferred == "chrome":
        return CHROME_BROWSER_NAMES
    if preferred:
        return (preferred,)
    return ()


def open_preferred_browser(
    url: str,
    *,
    preferred_browser: str | None = None,
    get_browser: Callable[[str], Any] | None = None,
    open_url: Callable[[str], Any] | None = None,
) -> dict[str, object]:
    """Open a URL with a preferred browser when possible, then fallback."""

    if get_browser is None or open_url is None:
        import webbrowser

        if get_browser is None:
            get_browser = webbrowser.get
        if open_url is None:
            open_url = webbrowser.open

    attempted: list[str] = []
    for name in browser_candidates(preferred_browser):
        attempted.append(name)
        try:
            get_browser(name).open(url)
            return {
                "opened": True,
                "browser": name,
                "fallback": False,
                "attempted": attempted,
            }
        except Exception:
            continue

    open_url(url)
    return {
        "opened": True,
        "browser": None,
        "fallback": True,
        "attempted": attempted,
    }


def schedule_browser_open(
    env: Mapping[str, str],
    url: str,
    *,
    delay_seconds: float = 1.0,
    timer_factory: Callable[[float, Callable[[], object]], Any] | None = None,
    opener: Callable[..., object] = open_preferred_browser,
) -> dict[str, object]:
    """Schedule browser opening when the runtime environment enables it."""

    browser = str(env.get("KQ_BROWSER", "")).strip().lower()
    if not should_auto_open_browser(env):
        return {
            "scheduled": False,
            "browser": browser,
            "delay_seconds": delay_seconds,
        }

    if timer_factory is None:
        import threading

        timer_factory = threading.Timer

    def _open() -> object:
        return opener(url, preferred_browser=browser)

    timer = timer_factory(delay_seconds, _open)
    timer.start()
    return {
        "scheduled": True,
        "browser": browser,
        "delay_seconds": delay_seconds,
    }


def run_server_with_browser_policy(
    server: Any,
    env: Mapping[str, str],
    url: str,
    *,
    output: Callable[[str], object] | None = print,
    schedule_browser: Callable[[Mapping[str, str], str], dict[str, object]] = schedule_browser_open,
    serve_runner: Callable[..., dict[str, bool]] = serve_until_interrupted,
) -> dict[str, object]:
    """Apply browser startup policy, then run the local server."""

    browser_schedule = schedule_browser(env, url)
    if not bool(browser_schedule.get("scheduled")) and output is not None:
        output(browser_autostart_disabled_message())
    serve_result = serve_runner(server, output=output)
    return {
        "browser": browser_schedule,
        "serve": serve_result,
    }
