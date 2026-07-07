from __future__ import annotations

import json

from kq_tool.data.report_crawler import (
    ReportSource,
    crawl_report_sources,
    create_sample_source_config,
    load_report_sources,
    parse_html_report,
    parse_rss_or_atom,
    render_market_report_markdown,
)


RSS = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel><title>Reports</title>
<item><title>경기 회복 보고서</title><link>https://example.com/a</link><description><![CDATA[경기 회복과 수출 회복이 이어진다. <b>원자재 강세</b>도 관찰된다.]]></description><pubDate>Wed, 01 Jul 2026 00:00:00 GMT</pubDate></item>
<item><title>물가 안정 리포트</title><link>https://example.com/b</link><description>물가 안정과 연착륙 기대가 높다.</description></item>
</channel></rss>"""


def test_parse_rss_or_atom_returns_compact_report_items() -> None:
    items = parse_rss_or_atom(RSS, source_name="테스트", source_url="https://example.com/rss", max_items=1)

    assert len(items) == 1
    assert items[0].title == "경기 회복 보고서"
    assert items[0].url == "https://example.com/a"
    assert "원자재 강세" in items[0].text
    assert "<b>" not in items[0].text


def test_parse_html_report_extracts_title_and_truncated_text() -> None:
    html = "<html><head><title>시장 시황</title></head><body><script>x()</script><p>경기 침체 우려와 안전자산 선호.</p></body></html>"

    item = parse_html_report(html, source_name="웹", source_url="https://example.com", max_chars_per_item=20)[0]

    assert item.title == "시장 시황"
    assert "경기 침체" in item.text
    assert "x()" not in item.text
    assert len(item.text) <= 20


def test_load_report_sources_filters_disabled_sources(tmp_path) -> None:
    config = tmp_path / "sources.json"
    config.write_text(
        json.dumps(
            [
                {"name": "A", "url": "https://a.example/rss", "kind": "rss", "enabled": True},
                {"name": "B", "url": "https://b.example/rss", "enabled": False},
            ]
        ),
        encoding="utf-8",
    )

    sources = load_report_sources(config)

    assert sources == [ReportSource(name="A", url="https://a.example/rss", kind="rss", enabled=True, max_items=None)]


def test_crawl_report_sources_uses_fetcher_without_network() -> None:
    sources = [ReportSource(name="테스트RSS", url="https://example.com/rss", kind="rss", max_items=2)]

    items, errors = crawl_report_sources(
        sources,
        fetcher=lambda url, **kwargs: RSS,
        check_robots=False,
    )

    assert errors == []
    assert [item.title for item in items] == ["경기 회복 보고서", "물가 안정 리포트"]


def test_render_market_report_markdown_keeps_source_urls_and_copyright_notice() -> None:
    items = parse_rss_or_atom(RSS, source_name="테스트", source_url="https://example.com/rss", max_items=1)

    md = render_market_report_markdown(items)

    assert "시장시황보고서 수집본" in md
    assert "원문 저작권" in md
    assert "https://example.com/a" in md
    assert "경기 회복" in md


def test_create_sample_source_config_writes_disabled_examples(tmp_path) -> None:
    path = create_sample_source_config(tmp_path / "report_sources.json")
    data = json.loads(path.read_text(encoding="utf-8"))

    assert data[0]["enabled"] is False
    assert data[0]["kind"] == "rss"
