from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_launcher_batch_points_to_windows_launcher() -> None:
    content = _read("RUN_KQ_TOOL.bat")

    assert "RUN_WINDOWS.ps1" in content
    assert "ExecutionPolicy Bypass" in content
    assert 'cd /d "%~dp0"' in content


def test_regression_batch_points_to_regression_script() -> None:
    content = _read("RUN_REGRESSION_CHECKS.bat")

    assert "RUN_REGRESSION_CHECKS.ps1" in content
    assert "ExecutionPolicy Bypass" in content
    assert 'cd /d "%~dp0"' in content


def test_regression_script_includes_documented_test_count_check() -> None:
    content = _read("RUN_REGRESSION_CHECKS.ps1")

    assert "Documented test count" in content
    assert "tests\\verify_test_count_docs.py" in content


def test_team_run_guide_documents_launcher_and_core_smoke() -> None:
    content = _read("docs/TEAM_RUN_GUIDE.md")

    assert "RUN_KQ_TOOL.bat" in content
    assert "http://127.0.0.1:8888/" in content
    assert "set PYTHONPATH=%CD%\\src;%PYTHONPATH%" in content
    assert "python -m kq_tool" in content
    assert "python tests\\smoke_api.py --include-stock --include-core --timeout 180" in content


def test_readme_links_handoff_and_regression_docs() -> None:
    content = _read("README.md")

    assert "docs/TEAM_RUN_GUIDE.md" in content
    assert "docs/final-regression-checks.md" in content
    assert "docs/release-readiness.md" in content


def test_windows_launcher_prefers_supported_python_before_314() -> None:
    content = _read("RUN_WINDOWS.ps1")

    python313 = content.index("Python313\\python.exe")
    python310 = content.index("Python310\\python.exe")
    python314 = content.index("Python314\\python.exe")

    assert python313 < python310 < python314
    assert 'py"; Args = @("-3.13")' in content
    assert 'py"; Args = @("-3.10")' in content
    assert 'py"; Args = @("-3")' in content


def test_windows_launcher_keeps_server_browser_autostart_disabled() -> None:
    content = _read("RUN_WINDOWS.ps1")

    assert '$env:KQ_AUTO_OPEN_BROWSER = "0"' in content
    assert '$env:KQ_DISABLE_TABPFN = "1"' in content
    assert "Open-BrowserLater" in content


def test_windows_launcher_sets_src_pythonpath_for_package_entrypoint() -> None:
    content = _read("RUN_WINDOWS.ps1")

    assert '$SrcPath = Join-Path $Root "src"' in content
    assert "$env:PYTHONPATH = $SrcPath" in content
    assert '$env:PYTHONPATH = "$SrcPath;$env:PYTHONPATH"' in content
    assert "& $python.Command @($python.Args) -m kq_tool" in content


def test_windows_launcher_prefers_chrome_before_default_browser() -> None:
    content = _read("RUN_WINDOWS.ps1")

    chrome_paths = content.index("$paths = @(")
    chrome_lookup = content.index("$chrome = $paths")
    chrome_launch = content.index("Start-Process -FilePath $chrome")
    fallback_launch = content.index("Start-Process $url")

    assert "Google\\Chrome\\Application\\chrome.exe" in content
    assert chrome_paths < chrome_lookup < chrome_launch < fallback_launch


def test_requirements_skips_hmmlearn_on_python_314() -> None:
    content = _read("requirements.txt")

    assert 'hmmlearn>=0.3.0; python_version < "3.14"' in content
