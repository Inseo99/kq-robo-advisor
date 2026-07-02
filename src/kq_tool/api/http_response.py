"""Low-level HTTP response helpers for the local server."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass

from .serialization import clean_json_value

JSON_CONTENT_TYPE = "application/json; charset=utf-8"

CORS_HEADERS = (
    ("Access-Control-Allow-Origin", "*"),
    ("Access-Control-Allow-Methods", "GET,OPTIONS"),
    ("Access-Control-Allow-Headers", "Content-Type"),
)

NO_CACHE_HEADERS = (
    ("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0"),
    ("Pragma", "no-cache"),
    ("Expires", "0"),
)

@dataclass(frozen=True)
class ResponseWriter:
    """Bound HTTP response callbacks for a BaseHTTPRequestHandler instance."""

    send_response: Callable[[int], object]
    send_header: Callable[[str, str], object]
    end_headers: Callable[[], object]
    write_body: Callable[[bytes], object]

    def empty(self, code: int) -> dict[str, object]:
        """Write an empty response with shared local headers."""

        return send_empty_response(
            send_response=self.send_response,
            send_header=self.send_header,
            end_headers=self.end_headers,
            code=code,
        )

    def body(self, body: bytes, content_type: str, *, code: int = 200) -> dict[str, object]:
        """Write a non-JSON body response with shared local headers."""

        return send_body_response(
            send_response=self.send_response,
            send_header=self.send_header,
            end_headers=self.end_headers,
            write_body=self.write_body,
            body=body,
            content_type=content_type,
            code=code,
        )

    def json(
        self,
        obj: object,
        *,
        code: int = 200,
        cleaner: Callable[[object], object] = clean_json_value,
    ) -> dict[str, object]:
        """Serialize and write a JSON response with shared local headers."""

        return send_json_response(
            obj,
            send_response=self.send_response,
            send_header=self.send_header,
            end_headers=self.end_headers,
            write_body=self.write_body,
            code=code,
            cleaner=cleaner,
        )


def make_response_writer(
    *,
    send_response: Callable[[int], object],
    send_header: Callable[[str, str], object],
    end_headers: Callable[[], object],
    write_body: Callable[[bytes], object],
) -> ResponseWriter:
    """Bind BaseHTTPRequestHandler callbacks into a reusable response writer."""

    return ResponseWriter(
        send_response=send_response,
        send_header=send_header,
        end_headers=end_headers,
        write_body=write_body,
    )


def cors_headers() -> tuple[tuple[str, str], ...]:
    """Return CORS headers used by every local response."""

    return CORS_HEADERS


def no_cache_headers() -> tuple[tuple[str, str], ...]:
    """Return no-cache headers used for local development responses."""

    return NO_CACHE_HEADERS


def content_headers(content_type: str, content_length: int) -> tuple[tuple[str, str], ...]:
    """Return content headers with a stringified length."""

    return (
        ("Content-Type", content_type),
        ("Content-Length", str(int(content_length))),
    )


def send_headers(
    send_header: Callable[[str, str], object],
    headers: tuple[tuple[str, str], ...],
) -> None:
    """Send a sequence of HTTP headers through a BaseHTTPRequestHandler callback."""

    for key, value in headers:
        send_header(key, value)


def send_empty_response(
    *,
    send_response: Callable[[int], object],
    send_header: Callable[[str, str], object],
    end_headers: Callable[[], object],
    code: int,
    cors_header_provider: Callable[[], tuple[tuple[str, str], ...]] = cors_headers,
) -> dict[str, object]:
    """Write an empty response with the local CORS policy."""

    send_response(code)
    send_headers(send_header, cors_header_provider())
    end_headers()
    return {"code": code}


def send_body_response(
    *,
    send_response: Callable[[int], object],
    send_header: Callable[[str, str], object],
    end_headers: Callable[[], object],
    write_body: Callable[[bytes], object],
    body: bytes,
    content_type: str,
    code: int = 200,
    content_header_provider: Callable[[str, int], tuple[tuple[str, str], ...]] = content_headers,
    cors_header_provider: Callable[[], tuple[tuple[str, str], ...]] = cors_headers,
    no_cache_header_provider: Callable[[], tuple[tuple[str, str], ...]] = no_cache_headers,
) -> dict[str, object]:
    """Write a body response with content, CORS, and no-cache headers."""

    send_response(code)
    send_headers(send_header, content_header_provider(content_type, len(body)))
    send_headers(send_header, cors_header_provider())
    send_headers(send_header, no_cache_header_provider())
    end_headers()
    write_body(body)
    return {"code": code, "content_type": content_type, "content_length": len(body)}


def json_response_body(
    obj: object,
    *,
    cleaner: Callable[[object], object] = clean_json_value,
) -> bytes:
    """Serialize a JSON response body using the project's JSON cleaner."""

    return json.dumps(cleaner(obj), ensure_ascii=False).encode("utf-8")


def send_json_response(
    obj: object,
    *,
    send_response: Callable[[int], object],
    send_header: Callable[[str, str], object],
    end_headers: Callable[[], object],
    write_body: Callable[[bytes], object],
    code: int = 200,
    cleaner: Callable[[object], object] = clean_json_value,
) -> dict[str, object]:
    """Serialize and write a JSON response using the shared response policy."""

    body = json_response_body(obj, cleaner=cleaner)
    return send_body_response(
        send_response=send_response,
        send_header=send_header,
        end_headers=end_headers,
        write_body=write_body,
        body=body,
        content_type=JSON_CONTENT_TYPE,
        code=code,
    )


def error_payload(error: BaseException | str) -> dict[str, str]:
    """Return the stable JSON error shape used by legacy API handlers."""

    return {"error": str(error)}


def send_json_action(
    action: Callable[[], object],
    *,
    send_json: Callable[[object, int], object],
    error_payload_fn: Callable[[BaseException | str], object] = error_payload,
    on_error: Callable[[BaseException], object] | None = None,
) -> dict[str, object]:
    """Run an action and send either its JSON result or a 500 JSON error."""

    try:
        payload = action()
    except Exception as exc:
        if on_error is not None:
            on_error(exc)
        send_json(error_payload_fn(exc), 500)
        return {"ok": False, "error": exc}

    send_json(payload, 200)
    return {"ok": True, "payload": payload}



