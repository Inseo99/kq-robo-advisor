from __future__ import annotations

import pytest

from kq_tool.api.static_files import read_static_file


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


def test_server_reuses_static_file_helper() -> None:
    import server

    assert server._kq_read_static_file is read_static_file
