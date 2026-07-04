from __future__ import annotations

import pandas as pd

from kq_tool.regime.macro_builder import REGIME_DEFINITION, build_macro_payload


def test_build_macro_payload_returns_defaults_without_excel_data() -> None:
    payload = build_macro_payload(None)

    assert payload["current"] == "리플레이션"
    assert payload["current_hint"] == "데이터 부족으로 추정값 사용"
    assert "골디락스" in payload["regimes"]
    assert payload["indicators"]["GDP_QoQ"] == 0.7


def test_build_macro_payload_extracts_macro_indicators_and_classifier_result() -> None:
    excel_macro = {
        "gdp": pd.DataFrame({"GDP 성장률": [0.004, 0.012]}),
        "rate": pd.DataFrame({"국고10년": [3.8, 4.1], "국고1년": [3.2, 3.6]}),
        "fx": pd.DataFrame({"미국 달러": [1300.0, 1365.5]}),
    }

    payload = build_macro_payload(
        excel_macro,
        classify_regime_fn=lambda _macro: {
            "regime": "스태그플레이션",
            "gdp": 1.2,
            "spread": 0.5,
        },
    )

    assert payload["current"] == "스태그플레이션"
    assert payload["indicators"]["GDP_QoQ"] == 1.2
    assert payload["indicators"]["장기금리10Y"] == 4.1
    assert payload["indicators"]["장단기스프레드"] == 0.5
    assert payload["indicators"]["환율USD"] == 1365.5
    assert "스태그플레이션 국면" in payload["current_hint"]


def test_build_macro_payload_falls_back_on_bad_macro_data() -> None:
    payload = build_macro_payload({"gdp": object()})

    assert payload["current"] == "리플레이션"
    assert payload["indicators"]["GDP_QoQ"] == 0.7


def test_server_reuses_macro_builder_helper() -> None:
    import server

    assert server._kq_build_macro_payload is build_macro_payload
    assert server._KQ_REGIME_DEFINITION is REGIME_DEFINITION
