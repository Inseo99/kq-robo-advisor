"""Startup and runtime health payload helpers."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path


def build_module_status(
    *,
    api_ready: bool,
    data_ready: bool,
    analyzer_ready: bool,
    portfolio_ready: bool,
    backtest_ready: bool,
    regime_ready: bool,
    screener_ready: bool,
) -> dict[str, bool]:
    """Build the module readiness section used by `/api/health`."""

    return {
        "api": bool(api_ready),
        "data": bool(data_ready),
        "analyzer": bool(analyzer_ready),
        "portfolio": bool(portfolio_ready),
        "backtest": bool(backtest_ready),
        "regime": bool(regime_ready),
        "screener": bool(screener_ready),
    }


def build_data_status(
    *,
    excel_data: object,
    excel_fin: object,
    excel_macro: object,
    cache_dir: object,
) -> dict[str, object]:
    """Build the data readiness section used by `/api/health`."""

    return {
        "excel_data": excel_data is not None,
        "excel_fin": excel_fin is not None,
        "excel_macro": excel_macro is not None,
        "cache_dir": cache_dir,
    }


def build_regime_status(regime_model: object) -> dict[str, object]:
    """Build the regime-model readiness section used by `/api/health`."""

    classifier = getattr(regime_model, "classifier", None) if regime_model is not None else None
    return {
        "trained": regime_model is not None and bool(getattr(regime_model, "trained", False)),
        "model_type": getattr(classifier, "model_type", None) if classifier is not None else None,
    }


def build_health_payload(
    *,
    base_dir: str | Path,
    modules: Mapping[str, bool],
    data_status: Mapping[str, object],
    universe_count: int,
    etf_count: int,
    regime_status: Mapping[str, object] | None = None,
) -> dict:
    """Build a compact health payload for local smoke checks."""

    base = Path(base_dir)
    modules = dict(modules)
    data_status = dict(data_status)
    regime_status = dict(regime_status or {})
    required_files = {
        "server.py": base / "server.py",
        "index.html": base / "index.html",
        "requirements.txt": base / "requirements.txt",
        "data/cache": base / "data" / "cache",
    }
    files = {name: path.exists() for name, path in required_files.items()}
    ok = (
        all(files.values())
        and all(bool(value) for value in modules.values())
        and int(universe_count) > 0
        and int(etf_count) > 0
    )
    return {
        "ok": bool(ok),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "app": "KQ Quant Tool",
        "files": files,
        "modules": modules,
        "data": data_status,
        "regime": regime_status,
        "counts": {
            "universe": int(universe_count),
            "etfs": int(etf_count),
        },
    }


def build_server_health_payload(
    *,
    base_dir: str | Path,
    module_flags: Mapping[str, bool],
    excel_data: object,
    excel_fin: object,
    excel_macro: object,
    cache_dir: object,
    regime_model: object,
    universe_count: int,
    etf_count: int,
) -> dict:
    """Build the full local-server health payload from runtime objects."""

    modules = build_module_status(
        api_ready=module_flags.get("api", False),
        data_ready=module_flags.get("data", False),
        analyzer_ready=module_flags.get("analyzer", False),
        portfolio_ready=module_flags.get("portfolio", False),
        backtest_ready=module_flags.get("backtest", False),
        regime_ready=module_flags.get("regime", False),
        screener_ready=module_flags.get("screener", False),
    )
    data_status = build_data_status(
        excel_data=excel_data,
        excel_fin=excel_fin,
        excel_macro=excel_macro,
        cache_dir=cache_dir,
    )
    return build_health_payload(
        base_dir=base_dir,
        modules=modules,
        data_status=data_status,
        universe_count=universe_count,
        etf_count=etf_count,
        regime_status=build_regime_status(regime_model),
    )
