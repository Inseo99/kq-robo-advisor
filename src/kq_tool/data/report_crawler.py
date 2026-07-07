"""Public market/analyst report collection helpers.

The crawler is intentionally conservative: it reads user-configured public RSS or
web URLs, respects robots.txt by default, and stores compact snippets for regime
context instead of copying full articles.
"""

from __future__ import annotations

import argparse
import datetime as dt
import html
import json
import re
import urllib.error
import urllib.parse
import urllib.request
import urllib.robotparser
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

DEFAULT_USER_AGENT = "KQQuantTool/0.1 (+local research; contact: user-configured)"
DEFAULT_CONFIG_PATH = "data/report_sources.json"
DEFAULT_OUTPUT_PATH = "data/market_reports/latest.md"
DEFAULT_MAX_CHARS_PER_ITEM = 1200
DEFAULT_MAX_ITEMS_PER_SOURCE = 5


@dataclass(frozen=True)
class ReportSource:
    """Configuration for one public report source."""

    name: str
    url: str
    kind: str = "rss"
    enabled: bool = True
    max_items: int | None = None


@dataclass(frozen=True)
class ReportItem:
    """A compact report item saved for qualitative regime context."""

    source: str
    title: str
    url: str
    text: str
    published: str | None = None


class _TextExtractor(HTMLParser):
    """Small HTML-to-text extractor without external dependencies."""

    def __init__(self) -> None:
        super().__init__()
        self._skip_depth = 0
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() in {"script", "style", "noscript", "svg"}:
            self._skip_depth += 1
        elif tag.lower() in {"p", "br", "li", "div", "section", "article", "h1", "h2", "h3"}:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in {"script", "style", "noscript", "svg"} and self._skip_depth:
            self._skip_depth -= 1
        elif tag.lower() in {"p", "li", "div", "section", "article"}:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skip_depth:
            self.parts.append(data)

    def text(self) -> str:
        return normalize_whitespace(" ".join(self.parts))


def normalize_whitespace(value: str) -> str:
    """Collapse noisy whitespace while preserving sentence readability."""

    value = html.unescape(str(value or ""))
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def strip_html(value: str) -> str:
    """Return compact readable text from an HTML fragment/document."""

    parser = _TextExtractor()
    parser.feed(str(value or ""))
    return parser.text()


def truncate_text(value: str, max_chars: int = DEFAULT_MAX_CHARS_PER_ITEM) -> str:
    """Truncate copied report text to a compact snippet."""

    text = normalize_whitespace(value)
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 1)].rstrip() + "…"


def load_report_sources(config_path: str | Path = DEFAULT_CONFIG_PATH) -> list[ReportSource]:
    """Load enabled report sources from JSON configuration."""

    path = Path(config_path)
    if not path.exists():
        return []
    raw = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(raw, list):
        raise ValueError("report source config must be a JSON list")

    sources: list[ReportSource] = []
    for index, row in enumerate(raw, 1):
        if not isinstance(row, Mapping):
            raise ValueError(f"source #{index} must be an object")
        if not bool(row.get("enabled", True)):
            continue
        name = normalize_whitespace(str(row.get("name") or f"source-{index}"))
        url = normalize_whitespace(str(row.get("url") or ""))
        if not url:
            raise ValueError(f"source #{index} is missing url")
        kind = str(row.get("kind") or "rss").strip().lower()
        if kind not in {"rss", "html", "text"}:
            raise ValueError(f"source #{index} kind must be rss, html, or text")
        max_items = row.get("max_items")
        sources.append(
            ReportSource(
                name=name,
                url=url,
                kind=kind,
                enabled=True,
                max_items=int(max_items) if max_items is not None else None,
            )
        )
    return sources


def fetch_url_text(
    url: str,
    *,
    timeout: float = 15.0,
    user_agent: str = DEFAULT_USER_AGENT,
) -> str:
    """Fetch a public URL as text using urllib only."""

    request = urllib.request.Request(url, headers={"User-Agent": user_agent})
    with urllib.request.urlopen(request, timeout=timeout) as response:  # nosec B310 - user configured public URLs
        data = response.read()
        content_type = response.headers.get("Content-Type", "")
    match = re.search(r"charset=([^;]+)", content_type, flags=re.I)
    encodings = [match.group(1).strip()] if match else []
    encodings += ["utf-8", "cp949", "euc-kr"]
    for encoding in encodings:
        try:
            return data.decode(encoding, errors="strict")
        except Exception:
            continue
    return data.decode("utf-8", errors="replace")


def robots_allows(
    url: str,
    *,
    user_agent: str = DEFAULT_USER_AGENT,
    timeout: float = 8.0,
) -> bool:
    """Return whether robots.txt allows fetching the configured URL."""

    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return False
    robots_url = urllib.parse.urlunparse((parsed.scheme, parsed.netloc, "/robots.txt", "", "", ""))
    parser = urllib.robotparser.RobotFileParser()
    parser.set_url(robots_url)
    try:
        parser.read()
    except Exception:
        # When robots.txt cannot be read, avoid blocking RSS-style user workflows.
        return True
    return parser.can_fetch(user_agent, url)


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].lower()


def _child_text(element: ET.Element, wanted: Sequence[str]) -> str | None:
    wanted_set = {name.lower() for name in wanted}
    for child in list(element):
        if _local_name(child.tag) in wanted_set:
            text = "".join(child.itertext())
            if text.strip():
                return text
    return None


def _link_text(element: ET.Element) -> str | None:
    for child in list(element):
        if _local_name(child.tag) != "link":
            continue
        href = child.attrib.get("href")
        if href:
            return href
        text = "".join(child.itertext()).strip()
        if text:
            return text
    return None


def parse_rss_or_atom(
    xml_text: str,
    *,
    source_name: str,
    source_url: str,
    max_items: int = DEFAULT_MAX_ITEMS_PER_SOURCE,
    max_chars_per_item: int = DEFAULT_MAX_CHARS_PER_ITEM,
) -> list[ReportItem]:
    """Parse RSS/Atom XML into compact report items."""

    root = ET.fromstring(xml_text.strip())
    entries = [el for el in root.iter() if _local_name(el.tag) in {"item", "entry"}]
    items: list[ReportItem] = []
    for entry in entries[:max_items]:
        title = normalize_whitespace(_child_text(entry, ["title"]) or "제목 없음")
        link = normalize_whitespace(_link_text(entry) or source_url)
        published = normalize_whitespace(
            _child_text(entry, ["pubDate", "published", "updated", "dc:date"]) or ""
        ) or None
        body = (
            _child_text(entry, ["description", "summary", "content", "encoded"])
            or _child_text(entry, ["title"])
            or ""
        )
        text = truncate_text(strip_html(body), max_chars_per_item)
        items.append(ReportItem(source=source_name, title=title, url=link, text=text, published=published))
    return items


def parse_html_report(
    html_text: str,
    *,
    source_name: str,
    source_url: str,
    max_chars_per_item: int = DEFAULT_MAX_CHARS_PER_ITEM,
) -> list[ReportItem]:
    """Parse one HTML page into a single compact report item."""

    title_match = re.search(r"<title[^>]*>(.*?)</title>", html_text, flags=re.I | re.S)
    title = strip_html(title_match.group(1)) if title_match else source_name
    text = truncate_text(strip_html(html_text), max_chars_per_item)
    return [ReportItem(source=source_name, title=title or source_name, url=source_url, text=text)]


def crawl_report_sources(
    sources: Iterable[ReportSource],
    *,
    fetcher: Callable[..., str] = fetch_url_text,
    check_robots: bool = True,
    user_agent: str = DEFAULT_USER_AGENT,
    timeout: float = 15.0,
    max_items_per_source: int = DEFAULT_MAX_ITEMS_PER_SOURCE,
    max_chars_per_item: int = DEFAULT_MAX_CHARS_PER_ITEM,
) -> tuple[list[ReportItem], list[dict[str, str]]]:
    """Fetch configured sources and return report items plus non-fatal errors."""

    items: list[ReportItem] = []
    errors: list[dict[str, str]] = []
    for source in sources:
        try:
            if check_robots and not robots_allows(source.url, user_agent=user_agent, timeout=timeout):
                errors.append({"source": source.name, "url": source.url, "error": "robots.txt disallows fetch"})
                continue
            text = fetcher(source.url, timeout=timeout, user_agent=user_agent)
            limit = source.max_items or max_items_per_source
            if source.kind == "rss":
                parsed = parse_rss_or_atom(
                    text,
                    source_name=source.name,
                    source_url=source.url,
                    max_items=limit,
                    max_chars_per_item=max_chars_per_item,
                )
            elif source.kind == "html":
                parsed = parse_html_report(
                    text,
                    source_name=source.name,
                    source_url=source.url,
                    max_chars_per_item=max_chars_per_item,
                )[:limit]
            else:
                parsed = [
                    ReportItem(
                        source=source.name,
                        title=source.name,
                        url=source.url,
                        text=truncate_text(text, max_chars_per_item),
                    )
                ]
            items.extend(parsed)
        except Exception as exc:  # keep other sources usable
            errors.append({"source": source.name, "url": source.url, "error": str(exc)})
    return items, errors


def render_market_report_markdown(
    items: Sequence[ReportItem],
    *,
    errors: Sequence[Mapping[str, str]] = (),
    generated_at: dt.datetime | None = None,
) -> str:
    """Render compact collected report snippets as the latest market report."""

    generated_at = generated_at or dt.datetime.now(dt.timezone.utc).astimezone()
    lines = [
        "# 시장시황보고서 수집본",
        "",
        f"생성시각: {generated_at:%Y-%m-%d %H:%M:%S %z}",
        "",
        "> 공개 RSS/웹페이지에서 국면 진단 참고용으로 일부 문장만 수집했습니다.",
        "> 원문 저작권은 각 제공자에게 있으며, 자세한 내용은 URL 원문을 확인하세요.",
        "",
    ]
    if not items:
        lines += ["수집된 리포트가 없습니다.", ""]
    for index, item in enumerate(items, 1):
        lines += [
            f"## {index}. {item.title}",
            f"- 출처: {item.source}",
            f"- URL: {item.url}",
        ]
        if item.published:
            lines.append(f"- 게시일: {item.published}")
        lines += ["", item.text or "(본문 요약 없음)", ""]
    if errors:
        lines += ["## 수집 오류", ""]
        for error in errors:
            lines.append(
                f"- {error.get('source', '?')}: {error.get('error', '')} ({error.get('url', '')})"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_market_report(
    items: Sequence[ReportItem],
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    *,
    errors: Sequence[Mapping[str, str]] = (),
) -> Path:
    """Write collected reports to the markdown file read by regime diagnostics."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_market_report_markdown(items, errors=errors), encoding="utf-8")
    return path


def create_sample_source_config(path: str | Path = DEFAULT_CONFIG_PATH) -> Path:
    """Create a disabled sample source config users can edit safely."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    sample = [
        {
            "name": "Example public RSS",
            "url": "https://example.com/rss.xml",
            "kind": "rss",
            "enabled": False,
            "max_items": 5,
        },
        {
            "name": "Example public market commentary page",
            "url": "https://example.com/market-commentary",
            "kind": "html",
            "enabled": False,
            "max_items": 1,
        },
    ]
    target.write_text(json.dumps(sample, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return target


def crawl_and_write_market_report(
    *,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    check_robots: bool = True,
    timeout: float = 15.0,
    max_chars_per_item: int = DEFAULT_MAX_CHARS_PER_ITEM,
) -> dict[str, object]:
    """Load configured sources, crawl them, and write latest market-report markdown."""

    sources = load_report_sources(config_path)
    items, errors = crawl_report_sources(
        sources,
        check_robots=check_robots,
        timeout=timeout,
        max_chars_per_item=max_chars_per_item,
    )
    path = write_market_report(items, output_path, errors=errors)
    return {"output": str(path), "items": len(items), "errors": errors, "sources": len(sources)}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect public market/analyst report snippets")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--init-sample", action="store_true", help="create a disabled sample config and exit")
    parser.add_argument("--no-robots", action="store_true", help="skip robots.txt check for internal/approved URLs")
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS_PER_ITEM)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.init_sample:
        path = create_sample_source_config(args.config)
        print(f"sample config written: {path}")
        return 0
    if not Path(args.config).exists():
        print(f"source config not found: {args.config}")
        print(f"create one with: python collect_market_reports.py --init-sample")
        return 2
    result = crawl_and_write_market_report(
        config_path=args.config,
        output_path=args.output,
        check_robots=not args.no_robots,
        timeout=args.timeout,
        max_chars_per_item=args.max_chars,
    )
    print(f"sources: {result['sources']}, items: {result['items']}, output: {result['output']}")
    for error in result["errors"]:  # type: ignore[index]
        print(f"WARN {error['source']}: {error['error']}")
    return 0


__all__ = [
    "DEFAULT_CONFIG_PATH",
    "DEFAULT_OUTPUT_PATH",
    "ReportItem",
    "ReportSource",
    "crawl_and_write_market_report",
    "crawl_report_sources",
    "create_sample_source_config",
    "load_report_sources",
    "parse_html_report",
    "parse_rss_or_atom",
    "render_market_report_markdown",
    "write_market_report",
]


if __name__ == "__main__":
    raise SystemExit(main())
