from __future__ import annotations

import re
from pathlib import Path

from kq_tool.screener.strategies import DISPLAY_NAME_QUALITY, SCREENER_DEFINITIONS


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INDEX_HTML = PROJECT_ROOT / "index.html"


def _index_html() -> str:
    return INDEX_HTML.read_text(encoding="utf-8")


def test_screener_filter_buttons_match_backend_order() -> None:
    html = _index_html()
    match = re.search(r'<div class="filter-row" id="scr-filters">(.*?)</div>', html, re.S)
    assert match is not None

    labels = re.findall(r"setFilter\('([^']+)'", match.group(1))

    assert labels == list(SCREENER_DEFINITIONS)[:6]


def test_strategy_backtest_buttons_match_strategy_metadata_order() -> None:
    html = _index_html()
    match = re.search(r'<span[^>]*>전략:</span>(.*?)</div>', html, re.S)
    assert match is not None

    strategy_keys = re.findall(r'data-bs="([^"]+)"', match.group(1))

    assert strategy_keys == ["all"] + list(SCREENER_DEFINITIONS)[:6]


def test_strategy_backtest_discloses_net_costs_and_app_approximation() -> None:
    html = _index_html()

    assert "무위험수익률" in html
    assert "비용 차감 후(net)" in html
    assert "앱 유니버스 기준 근사" in html
    assert "정본 수치는 검증 엔진(backtest-ksj" in html
    assert "전체 비교" in html
    assert "통합 성과표" in html


def test_screener_table_headers_keep_s2_after_reverse_dcf() -> None:
    html = _index_html()
    match = re.search(r"html\+=`<table class=\"screen-tbl\"><tr>(.*?)</tr>`;", html, re.S)
    assert match is not None

    headers = re.findall(r"<th>(.*?)</th>", match.group(1))

    assert headers == [
        "순위",
        "종목",
        "섹터",
        "현재가",
        "등락",
        "PER",
        "PBR",
        "ROE",
        "시총",
        "모멘텀",
        "S2",
        "신호",
    ]


def test_screener_diagnostics_show_kang_funnel_and_sector_column() -> None:
    html = _index_html()

    assert f"{DISPLAY_NAME_QUALITY} strict" in html
    assert "하위50%" in html
    assert "최종" in html
    assert "${v.sector || '-'}" in html

def test_market_report_panel_supports_upload_and_scroll() -> None:
    html = _index_html()

    assert 'id="mr-file"' in html
    assert 'id="mr-text"' in html
    assert 'registerMarketReport()' in html
    assert 'refreshMarketReport()' in html
    assert 'max-height:220px;overflow-y:auto' in html
    assert '/api/market_report' in html


