#!/usr/bin/env python3
"""Utilities for working with chess board inputs (FEN-first)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List
import re

from .fen_to_board_lp import load_fen_from_file, fen_to_facts


@dataclass
class BoardSpec:
    board_id: str
    fen_path: Path
    fen_text: str
    expected_mate: str

    def facts(self) -> List[str]:
        return fen_to_facts(self.fen_text)


def collect_board_specs(inputs: list[str]) -> list[BoardSpec]:
    """Resolve user inputs (files/directories) into canonical BoardSpec objects."""
    specs: list[BoardSpec] = []
    seen: set[Path] = set()

    def _try_add(path: Path) -> None:
        fen_path = _resolve_fen_path(path)
        if fen_path is None:
            return
        resolved = fen_path.resolve()
        if resolved in seen:
            return
        seen.add(resolved)
        fen_text = load_fen_from_file(resolved)
        expected = _extract_expected_from_fen(resolved)
        specs.append(BoardSpec(board_id=resolved.stem, fen_path=resolved, fen_text=fen_text, expected_mate=expected))

    for raw in inputs:
        path = Path(raw)
        if path.is_dir():
            for fen_file in sorted(path.glob("*.fen")):
                _try_add(fen_file)
        else:
            _try_add(path)
    return specs


def _resolve_fen_path(path: Path) -> Path | None:
    if not path.exists():
        print(f"Warning: '{path}' does not exist; skipping.")
        return None
    if path.is_dir():
        return None
    if path.suffix == ".fen":
        return path
    print(f"Warning: skipping '{path}' (expected .fen).")
    return None


def _extract_expected_from_fen(fen_path: Path) -> str:
    for line in fen_path.read_text().splitlines():
        if "Expected" in line:
            return line.lstrip("#").strip()
    return ""


def board_spec_to_cnl(spec: BoardSpec) -> str:
    """Convert a BoardSpec into the controlled natural language description."""
    facts = spec.facts()
    pieces: dict[str, list[str]] = {"white": [], "black": []}
    mover = "unknown"

    piece_pattern = re.compile(r"piece\((\w+),\s*(\w+),\s*(\d+),\s*(\d+)\)")
    king_pattern = re.compile(r"king\((\w+),\s*(\d+),\s*(\d+)\)")
    mover_pattern = re.compile(r"to_move\((\w+)\)")

    for fact in facts:
        if match := piece_pattern.match(fact.rstrip(".")):
            color, ptype, x, y = match.groups()
            pieces[color].append(f"{ptype} on ({x}, {y})")
            continue
        if match := king_pattern.match(fact.rstrip(".")):
            color, x, y = match.groups()
            pieces[color].append(f"king on ({x}, {y})")
            continue
        if match := mover_pattern.match(fact.rstrip(".")):
            mover = match.group(1)

    lines: List[str] = []
    for color in ("white", "black"):
        if pieces[color]:
            lines.append(f"{color.capitalize()} pieces: {', '.join(pieces[color])}.")
    lines.append(f"Side to move: {mover}.")
    lines.append("Board coordinates: files and ranks 1..8 (a=1, b=2, ..., h=8).")
    lines.append(
        "Input predicates available: piece(Color, Type, X, Y).  king(Color, X, Y).  to_move(Color)."
    )
    lines.append("Task: find a mate-in-one for the side to move.")
    return "\n".join(lines)


def board_spec_facts_text(spec: BoardSpec) -> str:
    return "\n".join(spec.facts()) + "\n"
