#!/usr/bin/env python3
"""Collect public market/analyst report snippets for regime diagnostics."""

from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from kq_tool.data.report_crawler import main


if __name__ == "__main__":
    raise SystemExit(main())
