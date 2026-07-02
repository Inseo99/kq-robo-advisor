from __future__ import annotations

from kq_tool.portfolio.recommender import (
    apply_signal_tilt,
    auto_regime_tilt,
    build_recommendation_report,
    load_regime_alpha_summary,
    probability_weighted_regime_target,
    regime_probability_blend,
    regime_alpha_signal_adjustment,
    default_asset_signal,
    portfolio_validity,
    signal_weight_multiplier,
    stock_analysis_to_asset_signal,
)


def test_regime_probability_blend_mixes_current_and_next_probs() -> None:
    result = regime_probability_blend(
        {
            "current": "골디락스",
            "probs": {"골디락스": 0.6, "디플레이션": 0.4},
            "next_quarter": {"골디락스": 0.2, "디플레이션": 0.8},
        }
    )

    assert round(result["골디락스"], 2) == 0.48
    assert round(result["디플레이션"], 2) == 0.52
    assert round(sum(result.values()), 6) == 1.0


def test_probability_weighted_regime_target_blends_asset_targets() -> None:
    targets = {
        "골디락스": {"STOCK": 0.8, "BOND": 0.2},
        "디플레이션": {"STOCK": 0.2, "BOND": 0.8},
    }

    result = probability_weighted_regime_target(
        {
            "current": "골디락스",
            "probs": {"골디락스": 1.0},
            "next_quarter": {"디플레이션": 1.0},
        },
        targets,
    )

    assert round(result["STOCK"], 2) == 0.62
    assert round(result["BOND"], 2) == 0.38
    assert round(sum(result.values()), 6) == 1.0

def test_auto_regime_tilt_uses_no_tilt_for_low_confidence() -> None:
    result = auto_regime_tilt({"current": "골디락스", "confidence": 0.3, "next_quarter": {}})

    assert result["regime_tilt"] == 0.0
    assert result["confidence_level"] == "무틸트"


def test_auto_regime_tilt_reduces_tilt_when_stability_is_low() -> None:
    result = auto_regime_tilt(
        {"current": "골디락스", "confidence": 0.9, "next_quarter": {"골디락스": 0.2}}
    )

    assert result["regime_tilt"] == 0.1
    assert result["stability_note"] == "국면 전환 임박, 틸트 약화"


def test_portfolio_validity_returns_bounded_days() -> None:
    result = portfolio_validity(
        {
            "current": "골디락스",
            "confidence": 0.8,
            "duration_avg": {"골디락스": 10},
            "next_quarter": {"골디락스": 0.7},
        }
    )

    assert result["valid_days"] == 90
    assert result["next_same_regime_prob"] == 70.0


def test_signal_weight_multiplier_responds_to_buy_and_sell() -> None:
    buy = signal_weight_multiplier({"cw_signal": "매수", "confidence": 0.8, "exit_days": 5})
    sell = signal_weight_multiplier({"cw_signal": "매도", "confidence": 0.8, "exit_days": 5})

    assert buy > 1.0
    assert sell < 1.0


def test_signal_weight_multiplier_accepts_robo_confidence_scale() -> None:
    ratio_scale = signal_weight_multiplier(
        {"cw_signal": "매수", "confidence": 0.8, "exit_days": 5}
    )
    point_scale = signal_weight_multiplier(
        {"cw_signal": "매수", "confidence": 8.0, "exit_days": 5}
    )
    mid_confidence = signal_weight_multiplier(
        {"cw_signal": "매수", "confidence": 5.0, "exit_days": 5}
    )

    assert point_scale == ratio_scale
    assert mid_confidence < point_scale


def test_default_asset_signal_is_neutral() -> None:
    signal = default_asset_signal(error="boom")

    assert signal["signal"] == "관망"
    assert signal["cw_signal"] == "관망"
    assert signal["error"] == "boom"


def test_stock_analysis_to_asset_signal_extracts_recommendation_fields() -> None:
    signal = stock_analysis_to_asset_signal(
        {
            "cur": 70000,
            "chg": 1.2,
            "cur_date": "2026-06-29",
            "robo": {
                "signal": "매수",
                "score": 72,
                "cw_signal": "매수",
                "cw_score": 68,
                "confidence": 0.8,
                "exit_days": 12,
                "ignored": "x",
            },
        }
    )

    assert signal == {
        "signal": "매수",
        "score": 72,
        "cw_signal": "매수",
        "cw_score": 68,
        "confidence": 0.8,
        "exit_days": 12,
        "cur": 70000,
        "chg": 1.2,
        "cur_date": "2026-06-29",
        "error": None,
    }


def test_apply_signal_tilt_normalizes_weights() -> None:
    weights, multipliers = apply_signal_tilt(
        {"A": 0.5, "B": 0.5},
        {"A": {"cw_signal": "매수", "confidence": 1.0, "exit_days": 10}},
    )

    assert round(sum(weights.values()), 6) == 1.0
    assert multipliers["A"] > multipliers["B"]


def test_build_recommendation_report_constructs_assets_and_counts() -> None:
    report = build_recommendation_report(
        snapshot={
            "current": "골디락스",
            "confidence": 0.8,
            "duration_avg": {"골디락스": 1.0},
            "next_quarter": {"골디락스": 0.7},
            "probs": {"골디락스": 0.8, "디플레이션": 0.2},
        },
        base_weights={"A": 0.7, "B": 0.3},
        regime_target={"A": 0.2, "C": 0.8},
        signal_map={
            "A": {"cw_signal": "매수", "confidence": 0.8, "exit_days": 10},
            "B": {"cw_signal": "관망"},
            "C": {"cw_signal": "매도", "confidence": 0.5, "exit_days": 4},
        },
        etf_meta={
            "A": ("Alpha", "주식", "공격"),
            "B": ("Bond", "채권", "방어"),
            "C": ("Commodity", "원자재", "인플레"),
        },
        components=[{"name": "base", "weight": 1.0}],
    )

    assert report["title"] == "국면 기반 완성형 추천 포트폴리오"
    assert round(sum(report["weights"].values()), 1) == 100.0
    assert {asset["ticker"] for asset in report["assets"]} == {"A", "B", "C"}
    assert report["action_counts"]["매수"] == 1
    assert report["action_counts"]["관망"] == 1
    assert report["action_counts"]["매도"] == 1
    assert "pre_signal_weights" in report["construction"]
    assert "regime_probability_blend" in report["construction"]
    assert report["construction"]["regime_target_source"] == "현재 국면 확률 70% + 다음 분기 국면 확률 30%"


def test_server_reuses_recommendation_signal_helpers() -> None:
    import server

    assert server._kq_default_asset_signal is default_asset_signal
    assert server._kq_stock_analysis_to_asset_signal is stock_analysis_to_asset_signal

def test_regime_alpha_signal_adjustment_weakens_negative_excess() -> None:
    result = regime_alpha_signal_adjustment(
        "디플레이션",
        [
            {
                "regime": "디플레이션",
                "events": "127",
                "n": "200",
                "edge_pct": "0.85",
                "random_pct": "1.18",
                "excess_pct": "-0.33",
                "p_value": "0.80",
            }
        ],
    )

    assert result["status"] == "약화"
    assert result["signal_tilt_scale"] < 1.0


def test_signal_weight_multiplier_respects_regime_alpha_scale() -> None:
    normal = signal_weight_multiplier({"cw_signal": "매수", "confidence": 1.0, "exit_days": 10})
    weakened = signal_weight_multiplier(
        {"cw_signal": "매수", "confidence": 1.0, "exit_days": 10},
        signal_tilt_scale=0.5,
    )

    assert weakened > 1.0
    assert weakened < normal


def test_load_regime_alpha_summary_reads_utf8_sig_csv(tmp_path) -> None:
    csv_path = tmp_path / "summary.csv"
    csv_path.write_text(
        "regime,events,edge_pct,random_pct,excess_pct,p_value\n골디락스,100,2.0,1.0,1.0,0.05\n",
        encoding="utf-8-sig",
    )

    rows = load_regime_alpha_summary(csv_path)

    assert rows[0]["regime"] == "골디락스"
    assert rows[0]["excess_pct"] == "1.0"


def test_build_recommendation_report_includes_regime_alpha_adjustment() -> None:
    report = build_recommendation_report(
        snapshot={
            "current": "디플레이션",
            "confidence": 0.8,
            "duration_avg": {"디플레이션": 1.0},
            "next_quarter": {"디플레이션": 0.7},
            "probs": {"디플레이션": 1.0},
        },
        base_weights={"A": 1.0},
        regime_target={"A": 1.0},
        signal_map={"A": {"cw_signal": "매수", "confidence": 1.0, "exit_days": 10}},
        etf_meta={"A": ("Alpha", "주식", "공격")},
        regime_alpha_summary=[
            {
                "regime": "디플레이션",
                "events": "100",
                "n": "200",
                "edge_pct": "0.5",
                "random_pct": "1.0",
                "excess_pct": "-0.5",
                "p_value": "0.9",
            }
        ],
    )

    adjustment = report["construction"]["regime_alpha_signal_adjustment"]
    assert adjustment["status"] == "약화"
    assert report["construction"]["signal_multipliers"]["A"] < 1.1


