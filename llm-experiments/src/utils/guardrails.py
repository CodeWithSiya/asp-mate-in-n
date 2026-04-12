#!/usr/bin/env python3
"""Helpers for optional syntax guardrail text shared across strategies."""
from __future__ import annotations

from pathlib import Path

DEFAULT_SYNTAX_GUARDRAIL_PATH = Path("prompts/syntax_guardrail.txt")
_CACHE: dict[Path, str] = {}


def load_syntax_guardrail(path: Path | None = None) -> str:
    target = path or DEFAULT_SYNTAX_GUARDRAIL_PATH
    if target not in _CACHE:
        _CACHE[target] = target.read_text().strip()
    return _CACHE[target]
