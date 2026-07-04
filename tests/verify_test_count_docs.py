"""Verify documented unit-test counts match pytest collection."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOCS = (
    ROOT / "README.md",
    ROOT / "docs" / "release-readiness.md",
)


def collect_unit_test_count() -> int:
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/unit", "--collect-only", "-q"],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    total = 0
    for line in result.stdout.splitlines():
        match = re.search(r":\s*(\d+)\s*$", line)
        if match:
            total += int(match.group(1))
    if total <= 0:
        raise RuntimeError("Could not determine unit-test count from pytest collect output.")
    return total


def documented_counts(path: Path) -> list[int]:
    text = path.read_text(encoding="utf-8")
    return [int(value) for value in re.findall(r"tests/unit`?\s*(?:기준)?\s*(\d+)개", text)]


def main() -> int:
    actual = collect_unit_test_count()
    errors: list[str] = []
    for path in DOCS:
        counts = documented_counts(path)
        if not counts:
            errors.append(f"{path.relative_to(ROOT)}: documented tests/unit count not found")
            continue
        wrong = [count for count in counts if count != actual]
        if wrong:
            errors.append(
                f"{path.relative_to(ROOT)}: documented {wrong}, actual {actual}"
            )

    if errors:
        print("Documented unit-test count mismatch:")
        for error in errors:
            print(f"- {error}")
        return 1

    print(f"Documented unit-test count OK: {actual}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
