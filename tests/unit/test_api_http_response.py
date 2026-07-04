from __future__ import annotations

import json

import numpy as np

from kq_tool.api.http_response import (
    JSON_CONTENT_TYPE,
    ResponseWriter,
    content_headers,
    cors_headers,
    error_payload,
    json_response_body,
    make_response_writer,
    no_cache_headers,
    send_body_response,
    send_empty_response,
    send_headers,
    send_json_action,
    send_json_response,
)


def test_cors_headers_match_local_api_policy() -> None:
    assert cors_headers() == (
        ("Access-Control-Allow-Origin", "*"),
        ("Access-Control-Allow-Methods", "GET,OPTIONS"),
        ("Access-Control-Allow-Headers", "Content-Type"),
    )


def test_no_cache_headers_match_local_development_policy() -> None:
    assert no_cache_headers() == (
        ("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0"),
        ("Pragma", "no-cache"),
        ("Expires", "0"),
    )


def test_content_headers_stringifies_content_length() -> None:
    assert content_headers("text/html; charset=utf-8", 12) == (
        ("Content-Type", "text/html; charset=utf-8"),
        ("Content-Length", "12"),
    )

def test_make_response_writer_binds_callbacks_and_writes_empty_body_and_json() -> None:
    calls: list[tuple[str, object]] = []
    writer = make_response_writer(
        send_response=lambda code: calls.append(("status", code)),
        send_header=lambda key, value: calls.append(("header", (key, value))),
        end_headers=lambda: calls.append(("end", None)),
        write_body=lambda data: calls.append(("body", data)),
    )

    assert isinstance(writer, ResponseWriter)
    assert writer.empty(204) == {"code": 204}
    body_result = writer.body(b"hello", "text/plain")
    json_result = writer.json({"ok": True}, code=201)

    assert body_result == {"code": 200, "content_type": "text/plain", "content_length": 5}
    assert json_result["code"] == 201
    assert json_result["content_type"] == "application/json; charset=utf-8"
    assert calls[0] == ("status", 204)
    assert ("body", b"hello") in calls
    assert ("body", b'{"ok": true}') in calls
def test_send_headers_applies_each_pair_in_order() -> None:
    sent: list[tuple[str, str]] = []

    send_headers(lambda key, value: sent.append((key, value)), (("A", "1"), ("B", "2")))

    assert sent == [("A", "1"), ("B", "2")]


def test_send_empty_response_writes_status_cors_and_end() -> None:
    calls: list[tuple[str, object]] = []

    result = send_empty_response(
        send_response=lambda code: calls.append(("status", code)),
        send_header=lambda key, value: calls.append(("header", (key, value))),
        end_headers=lambda: calls.append(("end", None)),
        code=204,
    )

    assert result == {"code": 204}
    assert calls[0] == ("status", 204)
    assert ("header", ("Access-Control-Allow-Origin", "*")) in calls
    assert calls[-1] == ("end", None)


def test_send_body_response_writes_content_cors_cache_headers_and_body() -> None:
    calls: list[tuple[str, object]] = []
    body = b"hello"

    result = send_body_response(
        send_response=lambda code: calls.append(("status", code)),
        send_header=lambda key, value: calls.append(("header", (key, value))),
        end_headers=lambda: calls.append(("end", None)),
        write_body=lambda data: calls.append(("body", data)),
        body=body,
        content_type="text/plain",
    )

    assert result == {"code": 200, "content_type": "text/plain", "content_length": 5}
    assert calls[0] == ("status", 200)
    assert ("header", ("Content-Type", "text/plain")) in calls
    assert ("header", ("Content-Length", "5")) in calls
    assert ("header", ("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")) in calls
    assert calls[-2:] == [("end", None), ("body", body)]


def test_send_json_response_serializes_and_writes_json_body() -> None:
    calls: list[tuple[str, object]] = []

    result = send_json_response(
        {"종목": "삼성전자", "x": np.float64(np.nan)},
        send_response=lambda code: calls.append(("status", code)),
        send_header=lambda key, value: calls.append(("header", (key, value))),
        end_headers=lambda: calls.append(("end", None)),
        write_body=lambda data: calls.append(("body", data)),
        code=201,
    )

    assert result["code"] == 201
    assert result["content_type"] == JSON_CONTENT_TYPE
    assert calls[0] == ("status", 201)
    body = calls[-1][1]
    assert isinstance(body, bytes)
    assert json.loads(body.decode("utf-8")) == {"종목": "삼성전자", "x": None}


def test_json_response_body_cleans_numpy_values_and_preserves_unicode() -> None:
    body = json_response_body({"종목": "삼성전자", "x": np.float64(np.nan)})

    assert json.loads(body.decode("utf-8")) == {"종목": "삼성전자", "x": None}


def test_error_payload_keeps_stable_error_shape() -> None:
    assert error_payload(ValueError("boom")) == {"error": "boom"}
    assert error_payload("plain message") == {"error": "plain message"}


def test_send_json_action_sends_success_payload() -> None:
    sent: list[tuple[object, int]] = []

    result = send_json_action(
        lambda: {"ok": True},
        send_json=lambda payload, code: sent.append((payload, code)),
    )

    assert result == {"ok": True, "payload": {"ok": True}}
    assert sent == [({"ok": True}, 200)]


def test_send_json_action_sends_error_payload_and_reports_error() -> None:
    sent: list[tuple[object, int]] = []
    errors: list[str] = []

    def action() -> object:
        raise ValueError("bad input")

    result = send_json_action(
        action,
        send_json=lambda payload, code: sent.append((payload, code)),
        on_error=lambda exc: errors.append(str(exc)),
    )

    assert result["ok"] is False
    assert isinstance(result["error"], ValueError)
    assert sent == [({"error": "bad input"}, 500)]
    assert errors == ["bad input"]


def test_server_reuses_http_response_helpers() -> None:
    import server

    assert server._kq_cors_headers is cors_headers
    assert server._kq_error_payload is error_payload
    assert server._kq_make_response_writer is make_response_writer
    assert server._kq_no_cache_headers is no_cache_headers
    assert server._kq_send_empty_response is send_empty_response
    assert server._kq_send_json_action is send_json_action
    assert server._kq_send_json_response is send_json_response


def test_server_handler_empty_response_sends_status_cors_and_headers() -> None:
    import server

    calls: list[tuple[str, object]] = []
    handler = object.__new__(server.Handler)
    handler.send_response = lambda code: calls.append(("status", code))
    handler.send_header = lambda key, value: calls.append(("header", (key, value)))
    handler.end_headers = lambda: calls.append(("end", None))

    handler._empty(404)

    assert calls[0] == ("status", 404)
    assert ("header", ("Access-Control-Allow-Origin", "*")) in calls
    assert calls[-1] == ("end", None)


def test_server_handler_json_action_uses_success_and_error_paths() -> None:
    import server

    sent: list[tuple[object, int]] = []
    handler = object.__new__(server.Handler)
    handler._json = lambda payload, code=200: sent.append((payload, code))

    handler._json_action(lambda: {"ok": True})
    handler._json_action(lambda: (_ for _ in ()).throw(ValueError("bad")))

    assert sent == [({"ok": True}, 200), ({"error": "bad"}, 500)]




