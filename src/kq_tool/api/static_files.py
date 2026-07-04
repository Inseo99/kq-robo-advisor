"""Static-file response helpers for the legacy HTTP server."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class StaticFilePayload:
    data: bytes
    content_type: str


def read_static_file(
    base_dir: str | Path,
    name: str,
    content_type: str,
    *,
    allowed_names: tuple[str, ...] = ("index.html",),
) -> StaticFilePayload:
    """Read an allowed static file from inside the project directory."""

    if name not in allowed_names:
        raise FileNotFoundError(name)

    base = Path(base_dir).resolve()
    path = (base / name).resolve()
    if base != path and base not in path.parents:
        raise FileNotFoundError(name)

    return StaticFilePayload(path.read_bytes(), content_type)

def serve_static_file(
    base_dir: str | Path,
    name: str,
    content_type: str,
    *,
    send_body: Callable[[bytes, str], object],
    send_empty: Callable[[int], object],
    read_file: Callable[[str | Path, str, str], StaticFilePayload] = read_static_file,
) -> dict[str, object]:
    """Read and send a static file, converting missing files to a 404 response."""

    try:
        payload = read_file(base_dir, name, content_type)
    except FileNotFoundError:
        send_empty(404)
        return {"ok": False, "code": 404, "name": name}

    send_body(payload.data, payload.content_type)
    return {
        "ok": True,
        "code": 200,
        "name": name,
        "content_type": payload.content_type,
        "content_length": len(payload.data),
    }

