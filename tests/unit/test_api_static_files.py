from __future__ import annotations

import pytest

from kq_tool.api.static_files import read_static_file, serve_static_file


def test_read_static_file_returns_allowed_file_payload(tmp_path) -> None:
    (tmp_path / "index.html").write_text("<html>ok</html>", encoding="utf-8")

    payload = read_static_file(tmp_path, "index.html", "text/html; charset=utf-8")

    assert payload.data == b"<html>ok</html>"
    assert payload.content_type == "text/html; charset=utf-8"


def test_read_static_file_rejects_files_outside_allowlist(tmp_path) -> None:
    (tmp_path / "server.py").write_text("secret", encoding="utf-8")

    with pytest.raises(FileNotFoundError):
        read_static_file(tmp_path, "server.py", "text/plain")


def test_read_static_file_rejects_path_traversal_even_if_allowed_name_is_broader(tmp_path) -> None:
    outside = tmp_path.parent / "outside.txt"
    outside.write_text("outside", encoding="utf-8")

    with pytest.raises(FileNotFoundError):
        read_static_file(
            tmp_path,
            "../outside.txt",
            "text/plain",
            allowed_names=("../outside.txt",),
        )


def test_serve_static_file_sends_allowed_file_payload(tmp_path) -> None:
    (tmp_path / "index.html").write_text("<html>ok</html>", encoding="utf-8")
    calls: list[tuple[str, object]] = []

    result = serve_static_file(
        tmp_path,
        "index.html",
        "text/html; charset=utf-8",
        send_body=lambda data, content_type: calls.append(("body", (data, content_type))),
        send_empty=lambda code: calls.append(("empty", code)),
    )

    assert result == {
        "ok": True,
        "code": 200,
        "name": "index.html",
        "content_type": "text/html; charset=utf-8",
        "content_length": 15,
    }
    assert calls == [("body", (b"<html>ok</html>", "text/html; charset=utf-8"))]


def test_serve_static_file_converts_missing_file_to_404(tmp_path) -> None:
    calls: list[tuple[str, object]] = []

    result = serve_static_file(
        tmp_path,
        "index.html",
        "text/html; charset=utf-8",
        send_body=lambda data, content_type: calls.append(("body", (data, content_type))),
        send_empty=lambda code: calls.append(("empty", code)),
    )

    assert result == {"ok": False, "code": 404, "name": "index.html"}
    assert calls == [("empty", 404)]

def test_server_reuses_static_file_helper() -> None:
    import server

    assert server._kq_serve_static_file is serve_static_file


