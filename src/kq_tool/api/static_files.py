"""Static-file response helpers for the legacy HTTP server."""

from __future__ import annotations

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
