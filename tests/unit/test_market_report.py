from __future__ import annotations

from kq_tool.regime.market_report import (
    analyze_market_report_text,
    build_market_report_context,
    list_market_report_items,
    parse_market_report_items,
    save_user_market_report,
)


def test_analyze_market_report_text_scores_and_aligns_regime_evidence() -> None:
    text = "경기 회복과 수출 회복이 이어지고 원자재 강세도 관찰된다. 위험선호는 제한적이다."

    result = analyze_market_report_text(
        text,
        current_regime="리플레이션",
        regime_order=["골디락스", "리플레이션", "스태그플레이션", "디플레이션"],
    )

    assert result["available"] is True
    assert result["dominant_regime"] == "리플레이션"
    assert result["alignment"] == "aligned"
    assert result["confidence"] > 50
    assert result["evidence"]["리플레이션"]


def test_analyze_market_report_text_flags_divergence_from_quant_regime() -> None:
    text = "경기 침체와 수요 부진이 커지고 안전자산 선호가 강해졌다."

    result = analyze_market_report_text(text, current_regime="골디락스")

    assert result["dominant_regime"] == "디플레이션"
    assert result["alignment"] == "diverged"
    assert "정량 국면(골디락스)과 차이" in result["summary"]


def test_build_market_report_context_reports_missing_file(tmp_path) -> None:
    result = build_market_report_context(tmp_path, current_regime="리플레이션")

    assert result["available"] is False
    assert result["source"] is None
    assert "정량 매크로 국면만 사용" in result["summary"]


def test_build_market_report_context_loads_first_available_report(tmp_path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    (data / "market_report.txt").write_text("물가 안정과 연착륙 기대가 높다.", encoding="utf-8")

    result = build_market_report_context(tmp_path, current_regime="골디락스")

    assert result["available"] is True
    assert result["source"] == "data/market_report.txt"
    assert result["dominant_regime"] == "골디락스"
    assert result["alignment"] == "aligned"

def test_parse_market_report_items_extracts_titles_urls_and_text() -> None:
    text = """# 시장시황보고서 수집본

## 1. 반도체 업황 점검
- 출처: 증권사A
- URL: https://example.com/report
- 게시일: 2026-07-01

경기 회복과 수출 회복이 이어진다.
"""

    items = parse_market_report_items(text, source_path="data/market_reports/latest.md")

    assert items == [
        {
            "title": "반도체 업황 점검",
            "source": "증권사A",
            "source_path": "data/market_reports/latest.md",
            "url": "https://example.com/report",
            "published": "2026-07-01",
            "text": "경기 회복과 수출 회복이 이어진다.",
        }
    ]


def test_save_user_market_report_appends_and_build_context_reads_reports(tmp_path) -> None:
    saved = save_user_market_report(
        tmp_path,
        title="내 시황 메모",
        source="내 리포트",
        url="https://example.com/mine",
        text="물가 안정과 연착륙 기대가 높다.",
    )

    assert saved["path"] == "data/market_reports/user_reports.md"
    items = list_market_report_items(tmp_path)
    assert items[0]["title"] == "내 시황 메모"
    assert items[0]["url"] == "https://example.com/mine"

    context = build_market_report_context(tmp_path, current_regime="골디락스")
    assert context["available"] is True
    assert context["source"] == "data/market_reports/user_reports.md"
    assert context["reports"][0]["title"] == "내 시황 메모"
    assert context["dominant_regime"] == "골디락스"

