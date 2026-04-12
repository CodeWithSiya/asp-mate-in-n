#!/usr/bin/env python3
"""Helpers for loading curated base ASP fragments for mate-in-one puzzles."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List
import re

BASE_DIR = Path("data/base")
PLACEMENT_RE = re.compile(r"placement\((\w+),\s*(\w+),\s*([\w\d_]+),\s*(\d+),\s*(\d+)\)\.")
TO_MOVE_RE = re.compile(r"to_move\((\w+)\)\.")
EXPECTED_RE = re.compile(r"expected\s+mate[s]?\s*:?\s*(.+)", re.IGNORECASE)


@dataclass(slots=True)
class BaseSpec:
    board_id: str
    base_path: Path
    base_text: str
    expected_mate: str = ""

    def facts(self) -> str:
        return self.base_text.strip()


def _collect_base_files(inputs: Iterable[str]) -> list[Path]:
    paths: list[Path] = []
    seen: set[Path] = set()
    raw_inputs = inputs or [str(BASE_DIR)]
    for raw in raw_inputs:
        candidate = Path(raw)
        if candidate.is_dir():
            for lp in sorted(candidate.glob("*_base.lp")):
                resolved = lp.resolve()
                if resolved not in seen:
                    paths.append(resolved)
                    seen.add(resolved)
        else:
            resolved = candidate.resolve()
            if resolved in seen:
                continue
            if resolved.suffix != ".lp":
                print(f"Warning: skipping '{candidate}' (expected .lp).")
                continue
            paths.append(resolved)
            seen.add(resolved)
    return paths


def _extract_expected(text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip().lstrip("%# ")
        match = EXPECTED_RE.search(stripped)
        if match:
            return match.group(1).strip()
    return ""


def load_base_specs(inputs: list[str]) -> list[BaseSpec]:
    files = _collect_base_files(inputs)
    specs: list[BaseSpec] = []
    for path in files:
        board_id = path.stem.replace("_base", "")
        text = path.read_text()
        expected = _extract_expected(text)
        specs.append(BaseSpec(board_id=board_id, base_path=path, base_text=text, expected_mate=expected))
    return specs


def base_spec_facts_text(spec: BaseSpec) -> str:
    return spec.base_text if spec.base_text.endswith("\n") else spec.base_text + "\n"


def base_spec_to_cnl(spec: BaseSpec) -> str:
    pieces: dict[str, List[str]] = {"white": [], "black": []}
    mover = "unknown"
    for match in PLACEMENT_RE.finditer(spec.base_text):
        color, piece, pid, row, col = match.groups()
        pieces[color].append(f"{piece} {pid} on ({col}, {row})")
    if pieces:
        for color in ("white", "black"):
            pieces[color].sort()
    if match := TO_MOVE_RE.search(spec.base_text):
        mover = match.group(1)

    lines: List[str] = []
    for color in ("white", "black"):
        if pieces[color]:
            lines.append(f"{color.capitalize()} pieces: {', '.join(pieces[color])}.")
    lines.append(f"Side to move: {mover}.")
    lines.append("Base fragment includes helper rules that derive legal_move/5 for white.")
    lines.append("Task: extend this base program into a full mate-in-one solver for the given position.")
    return "\n".join(lines)
