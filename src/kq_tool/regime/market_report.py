"""Market-report text helpers for qualitative regime diagnostics."""

from __future__ import annotations

import datetime as dt
import re
from pathlib import Path
from typing import Mapping

REGIME_REPORT_KEYWORDS: dict[str, tuple[tuple[str, float], ...]] = {
    "골디락스": (
        ("완만한 성장", 2.0),
        ("물가 안정", 2.0),
        ("연착륙", 1.8),
        ("soft landing", 1.8),
        ("실적 개선", 1.4),
        ("위험선호", 1.3),
        ("금리 인하 기대", 1.2),
        ("소비 회복", 1.1),
        ("성장 지속", 1.1),
    ),
    "리플레이션": (
        ("경기 회복", 2.0),
        ("수요 회복", 1.7),
        ("물가 반등", 1.6),
        ("원자재 강세", 1.5),
        ("인플레이션 반등", 1.5),
        ("제조업 개선", 1.3),
        ("수출 회복", 1.3),
        ("금리 상승", 1.1),
        ("reflation", 1.8),
    ),
    "스태그플레이션": (
        ("경기 둔화", 1.8),
        ("물가 상승", 1.8),
        ("고물가", 1.7),
        ("비용 부담", 1.4),
        ("마진 압박", 1.3),
        ("유가 급등", 1.5),
        ("금리 부담", 1.2),
        ("성장 둔화", 1.4),
        ("stagflation", 1.8),
    ),
    "디플레이션": (
        ("경기 침체", 2.0),
        ("수요 부진", 1.7),
        ("물가 둔화", 1.5),
        ("디스인플레이션", 1.4),
        ("deflation", 1.8),
        ("침체 우려", 1.6),
        ("안전자산 선호", 1.5),
        ("금리 인하", 1.2),
        ("장기채 강세", 1.4),
        ("실적 하향", 1.2),
    ),
}

USER_MARKET_REPORT_PATH = "data/market_reports/user_reports.md"
DEFAULT_MARKET_REPORT_PATHS = (
    USER_MARKET_REPORT_PATH,
    "data/market_report.txt",
    "data/market_report.md",
    "data/market_reports/latest.txt",
    "data/market_reports/latest.md",
    "docs/market_report.md",
    "market_report.txt",
)
MAX_REGISTERED_REPORT_CHARS = 50000
MAX_REPORT_ITEM_CHARS = 1600

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?。！？])\s+|[\r\n]+")
_HEADING_RE = re.compile(r"^##\s+(.+?)\s*$", flags=re.M)


def split_report_sentences(text: str) -> list[str]:
    """Split Korean/English market-report text into compact evidence sentences."""

    sentences = []
    for part in _SENTENCE_SPLIT_RE.split(str(text or "")):
        clean = " ".join(part.strip().split())
        if clean:
            sentences.append(clean)
    return sentences


def _read_text_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        return path.read_text(encoding="cp949", errors="replace")


def _compact_text(value: str, max_chars: int = MAX_REPORT_ITEM_CHARS) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 1)].rstrip() + "…"


def _clean_title(value: str) -> str:
    title = str(value or "").strip()
    title = re.sub(r"^\d+[.)]\s*", "", title)
    return title or "제목 없음"


def _parse_report_section(title: str, body: str, source_path: str) -> dict[str, object] | None:
    source_name = ""
    url = ""
    published = ""
    body_lines = []
    for line in body.splitlines():
        stripped = line.strip()
        if stripped.startswith("- 출처:"):
            source_name = stripped.split(":", 1)[1].strip()
        elif stripped.startswith("- URL:"):
            url = stripped.split(":", 1)[1].strip()
        elif stripped.startswith("- 게시일:"):
            published = stripped.split(":", 1)[1].strip()
        elif stripped.startswith("- ") and not body_lines:
            continue
        elif stripped:
            body_lines.append(stripped)
    text = _compact_text("\n".join(body_lines))
    if not text and not url:
        return None
    return {
        "title": _clean_title(title),
        "source": source_name or source_path,
        "source_path": source_path,
        "url": url,
        "published": published or None,
        "text": text,
    }


def parse_market_report_items(text: str, *, source_path: str = "local") -> list[dict[str, object]]:
    """Parse local markdown/text reports into compact UI items."""

    raw = str(text or "")
    matches = list(_HEADING_RE.finditer(raw))
    items: list[dict[str, object]] = []
    for index, match in enumerate(matches):
        title = match.group(1)
        if title.strip() in {"수집 오류"}:
            continue
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(raw)
        item = _parse_report_section(title, raw[start:end], source_path)
        if item is not None:
            items.append(item)
    if items:
        return items
    fallback = _compact_text(raw)
    if not fallback:
        return []
    return [
        {
            "title": Path(source_path).stem or "시장시황보고서",
            "source": source_path,
            "source_path": source_path,
            "url": "",
            "published": None,
            "text": fallback,
        }
    ]


def list_market_report_items(
    base_dir: str | Path,
    *,
    candidate_paths: tuple[str, ...] = DEFAULT_MARKET_REPORT_PATHS,
    max_items: int = 30,
) -> list[dict[str, object]]:
    """Return compact report items from all available local report files."""

    root = Path(base_dir)
    items: list[dict[str, object]] = []
    for relative in candidate_paths:
        path = root / relative
        if not path.exists() or not path.is_file():
            continue
        try:
            items.extend(parse_market_report_items(_read_text_file(path), source_path=relative))
        except Exception:
            continue
    return items[:max_items]


def analyze_market_report_text(
    text: str,
    *,
    current_regime: str | None = None,
    regime_order: list[str] | tuple[str, ...] | None = None,
    max_evidence: int = 3,
) -> dict[str, object]:
    """Return a qualitative regime read from a market-commentary report."""

    sentences = split_report_sentences(text)
    order = list(regime_order or REGIME_REPORT_KEYWORDS.keys())
    scores = {regime: 0.0 for regime in order}
    evidence: dict[str, list[str]] = {regime: [] for regime in order}
    lowered_sentences = [(sentence, sentence.lower()) for sentence in sentences]

    for regime in order:
        for keyword, weight in REGIME_REPORT_KEYWORDS.get(regime, ()):
            needle = keyword.lower()
            for sentence, lowered in lowered_sentences:
                if needle in lowered:
                    scores[regime] = scores.get(regime, 0.0) + float(weight)
                    if len(evidence.setdefault(regime, [])) < max_evidence and sentence not in evidence[regime]:
                        evidence[regime].append(sentence)

    total = sum(value for value in scores.values() if value > 0)
    normalized = {
        regime: (round(value / total, 4) if total > 0 else 0.0)
        for regime, value in scores.items()
    }
    dominant = max(scores, key=scores.get) if total > 0 else None
    confidence = normalized.get(dominant, 0.0) if dominant else 0.0

    alignment = "unavailable"
    if dominant and current_regime:
        alignment = "aligned" if dominant == current_regime else "diverged"
    elif dominant:
        alignment = "reference_only"

    summary = "시장시황보고서에서 국면 단서를 찾지 못했습니다."
    if dominant:
        summary = (
            f"시장시황보고서는 {dominant} 단서를 가장 강하게 지지합니다 "
            f"(정성 신뢰도 {confidence * 100:.1f}%)."
        )
        if current_regime:
            if dominant == current_regime:
                summary += " 정량 국면과 같은 방향입니다."
            else:
                summary += f" 정량 국면({current_regime})과 차이가 있어 보수적 점검이 필요합니다."

    return {
        "available": bool(str(text or "").strip()),
        "dominant_regime": dominant,
        "confidence": round(confidence * 100, 1),
        "alignment": alignment,
        "scores": {regime: round(value, 3) for regime, value in scores.items()},
        "normalized": normalized,
        "evidence": {regime: rows for regime, rows in evidence.items() if rows},
        "summary": summary,
    }


def load_market_report_text(
    base_dir: str | Path,
    *,
    candidate_paths: tuple[str, ...] = DEFAULT_MARKET_REPORT_PATHS,
) -> dict[str, object]:
    """Load and combine available local market-report text files."""

    root = Path(base_dir)
    checked = []
    loaded = []
    for relative in candidate_paths:
        path = root / relative
        checked.append(relative)
        if not path.exists() or not path.is_file():
            continue
        text = _read_text_file(path)
        loaded.append({"source": relative, "text": text})
    if not loaded:
        return {"available": False, "source": None, "sources": [], "text": "", "checked": checked}
    combined = "\n\n".join(f"# {row['source']}\n\n{row['text']}" for row in loaded)
    return {
        "available": True,
        "source": loaded[0]["source"],
        "sources": [row["source"] for row in loaded],
        "text": combined,
        "checked": checked,
    }


def save_user_market_report(
    base_dir: str | Path,
    *,
    title: str,
    text: str,
    url: str = "",
    source: str = "사용자 등록",
    output_path: str = USER_MARKET_REPORT_PATH,
) -> dict[str, object]:
    """Append a user-provided report to the local report file used by regime diagnostics."""

    clean_text = str(text or "").strip()
    clean_title = _clean_title(title)
    clean_url = str(url or "").strip()
    clean_source = str(source or "사용자 등록").strip() or "사용자 등록"
    if not clean_text:
        raise ValueError("보고서 본문이 비어 있습니다.")
    if len(clean_text) > MAX_REGISTERED_REPORT_CHARS:
        clean_text = clean_text[:MAX_REGISTERED_REPORT_CHARS].rstrip() + "\n\n(본문이 길어 일부만 저장됨)"

    path = Path(base_dir) / output_path
    path.parent.mkdir(parents=True, exist_ok=True)
    created_at = dt.datetime.now(dt.timezone.utc).astimezone()
    header = ""
    if not path.exists() or not path.read_text(encoding="utf-8", errors="replace").strip():
        header = (
            "# 사용자 등록 시장시황/애널리스트 리포트\n\n"
            "> 사용자가 로컬 분석 목적으로 직접 등록한 보고서입니다.\n\n"
        )
    lines = [
        f"## {clean_title}",
        f"- 출처: {clean_source}",
    ]
    if clean_url:
        lines.append(f"- URL: {clean_url}")
    lines += [f"- 게시일: {created_at:%Y-%m-%d %H:%M:%S %z}", "", clean_text, ""]
    with path.open("a", encoding="utf-8", newline="") as handle:
        if header:
            handle.write(header)
        handle.write("\n".join(lines))
    return {"ok": True, "path": output_path, "title": clean_title, "source": clean_source, "url": clean_url}


def build_market_report_context(
    base_dir: str | Path,
    *,
    current_regime: str | None = None,
    regime_order: list[str] | tuple[str, ...] | None = None,
) -> dict[str, object]:
    """Load and analyze local market-report commentary for regime context."""

    loaded = load_market_report_text(base_dir)
    reports = list_market_report_items(base_dir)
    if not loaded.get("available"):
        return {
            "available": False,
            "source": None,
            "sources": loaded.get("sources", []),
            "checked": loaded.get("checked", list(DEFAULT_MARKET_REPORT_PATHS)),
            "reports": reports,
            "summary": "시장시황보고서 파일이 없어 정량 매크로 국면만 사용합니다.",
        }

    analysis = analyze_market_report_text(
        str(loaded.get("text", "")),
        current_regime=current_regime,
        regime_order=regime_order,
    )
    analysis["source"] = loaded.get("source")
    analysis["sources"] = loaded.get("sources", [])
    analysis["checked"] = loaded.get("checked", [])
    analysis["reports"] = reports
    return analysis


_analyze_market_report_text = analyze_market_report_text
_build_market_report_context = build_market_report_context
_list_market_report_items = list_market_report_items
_save_user_market_report = save_user_market_report
