from __future__ import annotations

from kq_tool.api.health import (
    build_data_status,
    build_health_payload,
    build_module_status,
    build_regime_status,
    build_server_health_payload,
)


class _Classifier:
    model_type = "LightGBM"


class _RegimeModel:
    trained = True
    classifier = _Classifier()


def test_build_module_status_normalizes_flags() -> None:
    payload = build_module_status(
        api_ready=1,
        data_ready=True,
        analyzer_ready=False,
        portfolio_ready=True,
        backtest_ready=True,
        regime_ready=True,
        screener_ready=True,
    )

    assert payload["api"] is True
    assert payload["analyzer"] is False


def test_build_data_status_reports_loaded_excel_objects() -> None:
    payload = build_data_status(
        excel_data=object(),
        excel_fin=None,
        excel_macro={"macro": True},
        cache_dir="data/cache",
    )

    assert payload == {
        "excel_data": True,
        "excel_fin": False,
        "excel_macro": True,
        "cache_dir": "data/cache",
    }


def test_build_regime_status_reports_model_type_when_available() -> None:
    payload = build_regime_status(_RegimeModel())

    assert payload == {"trained": True, "model_type": "LightGBM"}


def test_build_health_payload_reports_ready_state(tmp_path) -> None:
    for name in ["server.py", "index.html", "requirements.txt"]:
        (tmp_path / name).write_text("", encoding="utf-8")
    (tmp_path / "data" / "cache").mkdir(parents=True)

    payload = build_health_payload(
        base_dir=tmp_path,
        modules={"data": True, "api": True},
        data_status={"excel_data": True},
        universe_count=10,
        etf_count=7,
        regime_status={"trained": True},
    )

    assert payload["ok"] is True
    assert payload["files"]["server.py"] is True
    assert payload["counts"] == {"universe": 10, "etfs": 7}
    assert payload["regime"]["trained"] is True


def test_build_health_payload_fails_when_required_file_missing(tmp_path) -> None:
    payload = build_health_payload(
        base_dir=tmp_path,
        modules={"data": True},
        data_status={},
        universe_count=1,
        etf_count=1,
    )

    assert payload["ok"] is False
    assert payload["files"]["server.py"] is False


def test_build_server_health_payload_combines_runtime_sections(tmp_path) -> None:
    for name in ["server.py", "index.html", "requirements.txt"]:
        (tmp_path / name).write_text("", encoding="utf-8")
    (tmp_path / "data" / "cache").mkdir(parents=True)

    payload = build_server_health_payload(
        base_dir=tmp_path,
        module_flags={
            "api": True,
            "data": True,
            "analyzer": True,
            "portfolio": True,
            "backtest": True,
            "regime": True,
            "screener": True,
        },
        excel_data=object(),
        excel_fin=object(),
        excel_macro=object(),
        cache_dir="data/cache",
        regime_model=_RegimeModel(),
        universe_count=10,
        etf_count=7,
    )

    assert payload["ok"] is True
    assert payload["data"]["excel_fin"] is True
    assert payload["regime"] == {"trained": True, "model_type": "LightGBM"}
