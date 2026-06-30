"""HTTP route dispatch helpers for the legacy server."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from urllib.parse import parse_qs, urlparse


from .routes import build_get_response


@dataclass(frozen=True)
class ApiResponse:
    kind: str
    payload: object
    code: int = 200
    content_type: str | None = None


def apply_api_response(
    response: ApiResponse,
    *,
    send_file: Callable[[object, str | None], object],
    send_json: Callable[[object, int], object],
    send_empty: Callable[[int], object],
) -> str:
    """Apply a dispatched API response through server-provided callbacks."""

    if response.kind == "file":
        send_file(response.payload, response.content_type)
        return "file"
    if response.kind == "json":
        send_json(response.payload, response.code)
        return "json"
    send_empty(response.code)
    return response.kind


def handle_dispatched_get(
    path_with_query: str,
    services: Mapping[str, Callable[..., object]],
    *,
    send_file: Callable[[object, str | None], object],
    send_json: Callable[[object, int], object],
    send_empty: Callable[[int], object],
) -> str:
    """Dispatch and apply a GET response through server-provided callbacks."""

    response = dispatch_get(path_with_query, services)
    return apply_api_response(
        response,
        send_file=send_file,
        send_json=send_json,
        send_empty=send_empty,
    )


def handle_dispatched_get_safely(
    path_with_query: str,
    services: Mapping[str, Callable[..., object]],
    *,
    send_file: Callable[[object, str | None], object],
    send_json: Callable[[object, int], object],
    send_empty: Callable[[int], object],
    send_error: Callable[[BaseException], object],
    on_error: Callable[[BaseException], object] | None = None,
) -> dict[str, object]:
    """Dispatch and apply a GET response, converting exceptions to errors."""

    try:
        kind = handle_dispatched_get(
            path_with_query,
            services,
            send_file=send_file,
            send_json=send_json,
            send_empty=send_empty,
        )
    except Exception as exc:
        if on_error is not None:
            on_error(exc)
        send_error(exc)
        return {"ok": False, "error": exc}
    return {"ok": True, "kind": kind}


def handle_legacy_get(
    path_with_query: str,
    *,
    send_file: Callable[[object, str | None], object],
    send_json: Callable[[object, int], object],
    send_empty: Callable[[int], object],
    stock: Callable[[object], object],
    screen: Callable[[], object],
    backtest: Callable[[], object],
    stratbt: Callable[[object], object],
    macro_payload: object,
    regime_ai: Callable[[], object],
    recommend_portfolio: Callable[[], object],
    health: Callable[[], object],
) -> str:
    """Apply the legacy fallback GET route table through server callbacks."""

    parsed = urlparse(path_with_query)
    query = parse_qs(parsed.query)
    path = parsed.path

    if path == "/":
        send_file("index.html", "text/html; charset=utf-8")
        return "file"
    if path == "/api/stock":
        stock(query)
        return "stock"
    if path == "/api/screen":
        screen()
        return "screen"
    if path == "/api/backtest":
        backtest()
        return "backtest"
    if path == "/api/stratbt":
        stratbt(query)
        return "stratbt"
    if path == "/api/macro":
        send_json(macro_payload, 200)
        return "macro"
    if path == "/api/regime_ai":
        regime_ai()
        return "regime_ai"
    if path == "/api/recommend_portfolio":
        recommend_portfolio()
        return "recommend_portfolio"
    if path == "/api/health":
        send_json(health(), 200)
        return "health"
    if path == "/api/ping":
        send_json({"ok": True}, 200)
        return "ping"
    send_empty(404)
    return "not_found"
def dispatch_get(path_with_query: str, services: Mapping[str, Callable[..., object]]) -> ApiResponse:
    """Route a GET path to a file or JSON response description."""

    return build_get_response(path_with_query, services, make_response=ApiResponse)


_dispatch_get = dispatch_get


