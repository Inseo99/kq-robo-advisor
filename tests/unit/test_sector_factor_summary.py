from pathlib import Path

from kq_tool.backtest.sector_factor_summary import build_sector_factor_summary
from kq_tool.screener.strategies import DISPLAY_NAME_QUALITY


ROOT = Path(__file__).resolve().parents[2]


def test_sector_factor_summary_uses_presentation_labels():
    payload = build_sector_factor_summary(str(ROOT))

    assert payload["ok"] is True
    rows = {row["code"]: row for row in payload["rows"]}
    assert rows["s1m_lsv"]["display_code"] == "M1 LSV"
    assert rows["s1m_kang"]["display_code"] == f"M1 {DISPLAY_NAME_QUALITY}"
    assert rows["s1m_kang"]["display_selection"] == DISPLAY_NAME_QUALITY


def test_sector_factor_summary_keeps_conditional_strategy_as_candidate_only():
    payload = build_sector_factor_summary(str(ROOT))

    assert payload["ok"] is True
    assert "후속 검증 후보" in payload["candidate"]
    assert "기존 국면 정의" in payload["candidate"]
    assert "수치는 게시하지 않고" in payload["candidate"]


def test_sector_factor_summary_exposes_one_sample_curve_when_available():
    payload = build_sector_factor_summary(str(ROOT))

    assert payload["ok"] is True
    curve = payload["sample_curve"]
    assert curve is not None
    assert curve["title"].startswith("M1")
    assert len(curve["dates"]) == len(curve["nav"])
    assert len(curve["dates"]) >= 2


