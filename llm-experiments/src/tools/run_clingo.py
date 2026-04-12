#!/usr/bin/env python3
"""Thin wrapper around utils.clingo_utils.run_clingo_program for manual runs."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from utils.clingo_utils import run_clingo_program, write_clingo_output
from utils.fen_to_board_lp import fen_to_facts, load_fen_from_file


DEFAULT_CONVERTED_BOARD = Path("data/boards/board_conversion.lp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Clingo against a generated ASP program.")
    parser.add_argument(
        "program",
        type=Path,
        help="Path to the ASP program to evaluate (e.g., results/<model>/<label>/generated.lp).",
    )
    parser.add_argument(
        "--fen-board",
        required=True,
        help="Path to a .fen file. It will be converted to ASP facts before running Clingo.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path for the solver transcript (default: alongside the ASP program).",
    )
    return parser.parse_args()


def verify_paths(paths: Iterable[str | Path]) -> list[str]:
    resolved: list[str] = []
    for candidate in paths:
        if candidate is None:
            continue
        path = Path(candidate)
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path}")
        resolved.append(str(path))
    return resolved


def convert_fen_to_board(fen_file: Path, destination: Path = DEFAULT_CONVERTED_BOARD) -> Path:
    if not fen_file.exists():
        raise FileNotFoundError(f"FEN file not found: {fen_file}")
    fen = load_fen_from_file(fen_file)
    facts = fen_to_facts(fen)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(facts) + "\n")
    print(f"Converted {fen_file} -> {destination}")
    return destination


def main() -> None:
    args = parse_args()
    verify_paths([args.program])
    board_path = convert_fen_to_board(Path(args.fen_board))
    asp_code = Path(args.program).read_text()
    board_facts = Path(board_path).read_text()
    result = run_clingo_program(asp_code, board_facts)
    output_path = args.output or Path(args.program).with_name("clingo_manual_run.txt")
    write_clingo_output(result, output_path)
    print(f"Wrote solver output to {output_path}")
    status = 0 if result.get("parsed", True) else 1
    sys.exit(status)


if __name__ == "__main__":
    main()
