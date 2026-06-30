from __future__ import annotations

import http.server
import socketserver

from kq_tool.api.runtime import (
    browser_autostart_disabled_message,
    browser_candidates,
    create_http_server,
    open_preferred_browser,
    run_server_with_browser_policy,
    schedule_browser_open,
    serve_until_interrupted,
    server_url,
    server_ready_messages,
    should_auto_open_browser,
    startup_intro_messages,
    ThreadingReusableHTTPServer,
    yfinance_status_messages,
)


class _FakeBrowser:
    def __init__(self, name: str, opened: list[tuple[str, str]]) -> None:
        self.name = name
        self.opened = opened

    def open(self, url: str) -> None:
        self.opened.append((self.name, url))


class _FakeTimer:
    def __init__(self, delay: float, callback) -> None:
        self.delay = delay
        self.callback = callback
        self.started = False

    def start(self) -> None:
        self.started = True
        self.callback()


class _FakeServer:
    instances: list["_FakeServer"] = []

    def __init__(self, address=None, handler=None, *, interrupt: bool = False) -> None:
        self.address = address
        self.handler = handler
        self.interrupt = interrupt
        self.calls = 0
        _FakeServer.instances.append(self)

    @classmethod
    def reset(cls) -> None:
        cls.instances = []

    def serve_forever(self) -> None:
        self.calls += 1
        if self.interrupt:
            raise KeyboardInterrupt


class _InterruptingServer(_FakeServer):
    def __init__(self, *, interrupt: bool = False) -> None:
        super().__init__(interrupt=interrupt)


class _FakeHandler:
    pass


def test_server_url_uses_http_loopback_shape() -> None:
    assert server_url("127.0.0.1", "8888") == "http://127.0.0.1:8888"


def test_threading_reusable_http_server_keeps_local_server_policy() -> None:
    assert issubclass(ThreadingReusableHTTPServer, socketserver.ThreadingMixIn)
    assert issubclass(ThreadingReusableHTTPServer, http.server.HTTPServer)
    assert ThreadingReusableHTTPServer.daemon_threads is True
    assert ThreadingReusableHTTPServer.allow_reuse_address is True


def test_create_http_server_passes_address_and_handler() -> None:
    _FakeServer.reset()

    server = create_http_server(_FakeServer, _FakeHandler, host="127.0.0.1", port="8888")

    assert server is _FakeServer.instances[0]
    assert server.address == ("127.0.0.1", 8888)
    assert server.handler is _FakeHandler


def test_startup_intro_messages_keep_check_before_network_probe() -> None:
    assert startup_intro_messages() == (
        "\n=== KQ Quant Tool ===",
        "Yahoo Finance 연결 확인 중...",
    )


def test_yfinance_status_messages_cover_connected_and_fallback() -> None:
    assert yfinance_status_messages(True) == ("OK 연결 성공 - 실시간 데이터로 동작합니다",)

    fallback = yfinance_status_messages(False)
    assert fallback[0] == "WARN 연결 실패 - 샘플(시뮬레이션) 데이터로 동작합니다"
    assert "finance.yahoo.com" in fallback[1]
    assert "pip install --upgrade yfinance curl_cffi" in fallback[2]


def test_server_ready_messages_keep_manual_browser_guidance() -> None:
    assert server_ready_messages("http://127.0.0.1:8888") == (
        "http://127.0.0.1:8888",
        "Chrome에서 위 주소를 열어 사용하세요.",
        "Ctrl+C 로 종료\n",
    )


def test_browser_autostart_disabled_message_keeps_edge_context() -> None:
    assert browser_autostart_disabled_message() == (
        "브라우저 자동 열기 꺼짐: Edge가 뜨지 않도록 기본값을 변경했습니다."
    )


def test_serve_until_interrupted_reports_normal_return() -> None:
    server = _FakeServer()

    result = serve_until_interrupted(server, output=None)

    assert result == {"interrupted": False}
    assert server.calls == 1


def test_serve_until_interrupted_prints_ctrl_c_message() -> None:
    server = _InterruptingServer(interrupt=True)
    lines: list[str] = []

    result = serve_until_interrupted(server, output=lines.append)

    assert result == {"interrupted": True}
    assert server.calls == 1
    assert lines == ["\n종료"]


def test_should_auto_open_browser_is_off_by_default() -> None:
    assert should_auto_open_browser({}) is False
    assert should_auto_open_browser({"KQ_AUTO_OPEN_BROWSER": "0"}) is False


def test_should_auto_open_browser_accepts_explicit_truthy_values() -> None:
    assert should_auto_open_browser({"KQ_AUTO_OPEN_BROWSER": "1"}) is True
    assert should_auto_open_browser({"KQ_AUTO_OPEN_BROWSER": "true"}) is True
    assert should_auto_open_browser({"KQ_AUTO_OPEN_BROWSER": "YES"}) is True


def test_browser_candidates_expands_chrome_preference() -> None:
    assert browser_candidates("chrome") == (
        "chrome",
        "google-chrome",
        "chromium",
        "chromium-browser",
    )


def test_open_preferred_browser_uses_first_working_named_browser() -> None:
    opened: list[tuple[str, str]] = []
    attempted: list[str] = []

    def get_browser(name: str) -> _FakeBrowser:
        attempted.append(name)
        if name == "chrome":
            raise RuntimeError("not registered")
        return _FakeBrowser(name, opened)

    result = open_preferred_browser(
        "http://127.0.0.1:8888/",
        preferred_browser="chrome",
        get_browser=get_browser,
        open_url=lambda url: opened.append(("fallback", url)),
    )

    assert attempted == ["chrome", "google-chrome"]
    assert opened == [("google-chrome", "http://127.0.0.1:8888/")]
    assert result["browser"] == "google-chrome"
    assert result["fallback"] is False


def test_open_preferred_browser_falls_back_when_no_preference_is_set() -> None:
    opened: list[tuple[str, str]] = []

    result = open_preferred_browser(
        "http://127.0.0.1:8888/",
        preferred_browser="",
        get_browser=lambda name: _FakeBrowser(name, opened),
        open_url=lambda url: opened.append(("fallback", url)),
    )

    assert opened == [("fallback", "http://127.0.0.1:8888/")]
    assert result["browser"] is None
    assert result["fallback"] is True
    assert result["attempted"] == []


def test_schedule_browser_open_skips_when_disabled() -> None:
    timers: list[_FakeTimer] = []

    result = schedule_browser_open(
        {"KQ_AUTO_OPEN_BROWSER": "0", "KQ_BROWSER": "chrome"},
        "http://127.0.0.1:8888",
        timer_factory=lambda delay, callback: timers.append(_FakeTimer(delay, callback)) or timers[-1],
        opener=lambda *args, **kwargs: None,
    )

    assert result["scheduled"] is False
    assert result["browser"] == "chrome"
    assert timers == []


def test_schedule_browser_open_uses_timer_and_preferred_browser() -> None:
    timers: list[_FakeTimer] = []
    opened: list[tuple[str, str]] = []

    def opener(url: str, *, preferred_browser: str | None = None):
        opened.append((preferred_browser or "", url))

    result = schedule_browser_open(
        {"KQ_AUTO_OPEN_BROWSER": "1", "KQ_BROWSER": "chrome"},
        "http://127.0.0.1:8888",
        delay_seconds=2.5,
        timer_factory=lambda delay, callback: timers.append(_FakeTimer(delay, callback)) or timers[-1],
        opener=opener,
    )

    assert result == {"scheduled": True, "browser": "chrome", "delay_seconds": 2.5}
    assert len(timers) == 1
    assert timers[0].started is True
    assert timers[0].delay == 2.5
    assert opened == [("chrome", "http://127.0.0.1:8888")]


def test_run_server_with_browser_policy_prints_disabled_message_before_serving() -> None:
    server = _FakeServer()
    lines: list[str] = []

    result = run_server_with_browser_policy(
        server,
        {"KQ_AUTO_OPEN_BROWSER": "0"},
        "http://127.0.0.1:8888",
        output=lines.append,
    )

    assert result == {
        "browser": {"scheduled": False, "browser": "", "delay_seconds": 1.0},
        "serve": {"interrupted": False},
    }
    assert server.calls == 1
    assert lines == [browser_autostart_disabled_message()]


def test_run_server_with_browser_policy_skips_disabled_message_when_scheduled() -> None:
    server = _FakeServer()
    lines: list[str] = []

    result = run_server_with_browser_policy(
        server,
        {"KQ_AUTO_OPEN_BROWSER": "1"},
        "http://127.0.0.1:8888",
        output=lines.append,
        schedule_browser=lambda env, url: {
            "scheduled": True,
            "browser": "chrome",
            "delay_seconds": 1.0,
        },
    )

    assert result == {
        "browser": {"scheduled": True, "browser": "chrome", "delay_seconds": 1.0},
        "serve": {"interrupted": False},
    }
    assert server.calls == 1
    assert lines == []


def test_server_reuses_runtime_helpers() -> None:
    import server

    assert issubclass(server.KQServer, ThreadingReusableHTTPServer)
    assert server._kq_browser_autostart_disabled_message is browser_autostart_disabled_message
    assert server._kq_create_http_server is create_http_server
    assert server._kq_run_server_with_browser_policy is run_server_with_browser_policy
    assert server._kq_schedule_browser_open is schedule_browser_open
    assert server._kq_serve_until_interrupted is serve_until_interrupted
    assert server._kq_server_url is server_url
    assert server._kq_server_ready_messages is server_ready_messages
    assert server._kq_should_auto_open_browser is should_auto_open_browser
    assert server._kq_startup_intro_messages is startup_intro_messages
    assert server._kq_open_preferred_browser is open_preferred_browser
    assert server._kq_yfinance_status_messages is yfinance_status_messages
