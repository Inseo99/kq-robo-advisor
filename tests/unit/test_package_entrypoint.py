from __future__ import annotations

import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_package_main_delegates_to_legacy_server_main(monkeypatch) -> None:
    from kq_tool.__main__ import main

    calls: list[str] = []
    fake_server = types.SimpleNamespace(main=lambda: calls.append("main"))
    monkeypatch.setitem(sys.modules, "server", fake_server)

    main()

    assert calls == ["main"]


def test_server_py_keeps_compatibility_main_entrypoint() -> None:
    content = (ROOT / "server.py").read_text(encoding="utf-8")

    assert "\ndef main():" in content
    assert "if __name__ == '__main__':\n    main()" in content


def test_pyproject_exposes_console_script_entrypoint() -> None:
    content = (ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert "[project.scripts]" in content
    assert 'kq-tool = "kq_tool.__main__:main"' in content
