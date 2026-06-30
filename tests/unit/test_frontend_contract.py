from __future__ import annotations

import re
from pathlib import Path

from kq_tool.backtest.strategy_meta import QUANT, QUANT_COMPARE, QUANT_S2, ROBO
from kq_tool.screener.strategies import SCREENER_DEFINITIONS


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INDEX_HTML = PROJECT_ROOT / "index.html"


def _index_html() -> str:
    return INDEX_HTML.read_text(encoding="utf-8")


def test_screener_filter_buttons_match_backend_order() -> None:
    html = _index_html()
    match = re.search(r'<div class="filter-row" id="scr-filters">(.*?)</div>', html, re.S)
    assert match is not None

    labels = re.findall(r"setFilter\('([^']+)'", match.group(1))

    assert labels == ["all", *list(SCREENER_DEFINITIONS)[:6]]


def test_strategy_backtest_buttons_match_strategy_metadata_order() -> None:
    html = _index_html()
    match = re.search(r'<span[^>]*>전략:</span>(.*?)</div>', html, re.S)
    assert match is not None

    strategy_keys = re.findall(r'data-bs="([^"]+)"', match.group(1))

    assert strategy_keys == [QUANT_COMPARE.key, QUANT.key, QUANT_S2.key, ROBO.key]


def test_screener_table_headers_keep_s2_after_reverse_dcf() -> None:
    html = _index_html()
    match = re.search(r"let html=`<table class=\"screen-tbl\"><tr>(.*?)</tr>`;", html, re.S)
    assert match is not None

    headers = re.findall(r"<th>(.*?)</th>", match.group(1))

    assert headers == [
        "종목",
        "현재가",
        "등락",
        "PER",
        "PBR",
        "ROE",
        "모멘텀",
        "내재성장률",
        "S2",
        "신호",
    ]
