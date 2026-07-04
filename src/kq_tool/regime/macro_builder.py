"""Macro regime payload builder."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy


REGIME_DEFINITION: dict[str, dict] = {
    "골디락스": {
        "성장": "↑",
        "물가": "↓",
        "주식": 8,
        "국채": 2,
        "원자재": -1,
        "금": -2,
        "현금": 1,
        "label": "성장↑·물가↓",
        "color": "#3fb950",
        "hint": "주식·IT 강세, 채권 중립",
    },
    "리플레이션": {
        "성장": "↑",
        "물가": "↑",
        "주식": 6,
        "국채": -3,
        "원자재": 9,
        "금": 3,
        "현금": 1,
        "label": "성장↑·물가↑",
        "color": "#d29922",
        "hint": "원자재·에너지 수혜, 채권 약세",
    },
    "스태그플레이션": {
        "성장": "↓",
        "물가": "↑",
        "주식": -7,
        "국채": -4,
        "원자재": 7,
        "금": 6,
        "현금": 1,
        "label": "성장↓·물가↑",
        "color": "#f85149",
        "hint": "금·원자재 헤지, 현금 방어",
    },
    "디플레이션": {
        "성장": "↓",
        "물가": "↓",
        "주식": -6,
        "국채": 7,
        "원자재": -5,
        "금": 1,
        "현금": 1,
        "label": "성장↓·물가↓",
        "color": "#388bfd",
        "hint": "국채 강세, 방어주·현금 선호",
    },
}

DEFAULT_MACRO_INDICATORS: dict[str, float] = {
    "PMI": 52.1,
    "CPI_YoY": 2.8,
    "GDP_QoQ": 0.7,
    "기준금리": 3.50,
    "장기금리10Y": 3.85,
    "장단기스프레드": 0.35,
    "VIX": 18.2,
}


def build_macro_payload(
    excel_macro: Mapping[str, object] | None,
    *,
    classify_regime_fn: Callable[[Mapping[str, object]], dict] | None = None,
) -> dict:
    """Build the macro regime payload used by the legacy UI/API."""

    indicators = dict(DEFAULT_MACRO_INDICATORS)
    current = "리플레이션"
    hint = "데이터 부족으로 추정값 사용"

    if excel_macro:
        try:
            if "gdp" in excel_macro:
                gdp_df = excel_macro["gdp"]
                growth_columns = [col for col in gdp_df.columns if "성장률" in col]
                if growth_columns:
                    indicators["GDP_QoQ"] = round(
                        float(gdp_df[growth_columns[0]].dropna().iloc[-1]) * 100,
                        2,
                    )

            if "rate" in excel_macro:
                rate_df = excel_macro["rate"]
                long_columns = [col for col in rate_df.columns if "국고10년" in col]
                short_columns = [col for col in rate_df.columns if "국고1년" in col]
                if long_columns:
                    indicators["장기금리10Y"] = round(
                        float(rate_df[long_columns[0]].dropna().iloc[-1]),
                        2,
                    )
                if short_columns and long_columns:
                    spread = float(rate_df[long_columns[0]].dropna().iloc[-1]) - float(
                        rate_df[short_columns[0]].dropna().iloc[-1]
                    )
                    indicators["장단기스프레드"] = round(spread, 2)

            if "fx" in excel_macro:
                fx_df = excel_macro["fx"]
                usd_columns = [col for col in fx_df.columns if "미국" in col or "달러" in col]
                if usd_columns:
                    indicators["환율USD"] = round(
                        float(fx_df[usd_columns[0]].dropna().iloc[-1]),
                        1,
                    )

            if classify_regime_fn is not None:
                regime_result = classify_regime_fn(excel_macro)
                current = regime_result.get("regime", "리플레이션")
                hint = (
                    f"GDP {regime_result.get('gdp', 0)}% + "
                    f"장단기스프레드 {regime_result.get('spread', 0)}%p → {current} 국면"
                )
        except Exception:
            pass

    return {
        "regimes": deepcopy(REGIME_DEFINITION),
        "indicators": indicators,
        "current": current,
        "current_hint": hint,
    }


_build_macro_payload = build_macro_payload
