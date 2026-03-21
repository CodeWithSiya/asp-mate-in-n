#!/usr/bin/env python3
"""Thin wrapper to run Clingo on generated ASP programs plus board facts."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from utils.fen_to_board_lp import fen_to_facts, load_fen_from_file


DEFAULT_CONVERTED_BOARD = Path("data/boards/board_conversion.lp")


def parse_args() -> tuple[argparse.Namespace, list[str]]:
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
        "--clingo-path",
        default=os.environ.get("CLINGO_BIN", "clingo"),
        help="Path to the clingo binary. Defaults to $CLINGO_BIN or 'clingo'.",
    )
    parser.add_argument(
        "--models",
        type=int,
        default=0,
        help="Forwarded to `--models` (0 means enumerate all stable models).",
    )
    parser.add_argument(
        "--threads",
        type=int,
        help="Forwarded to `--threads`. Omit to accept Clingo's default.",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Include Clingo's `--stats` output.",
    )
    args, extras = parser.parse_known_args()
    return args, extras


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


def build_command(args: argparse.Namespace, board_path: Path) -> list[str]:
    cmd = [args.clingo_path, str(args.program), str(board_path)]
    cmd.extend(["--models", str(args.models)])
    if args.threads:
        cmd.extend(["--threads", str(args.threads)])
    if args.stats:
        cmd.append("--stats")
    return cmd


def append_extras(cmd: list[str], extras: list[str]) -> None:
    if not extras:
        return
    cmd.extend(extras)


def run_clingo(cmd: list[str]) -> int:
    print("Running:", " ".join(cmd))
    completed = subprocess.run(cmd, check=False)
    return completed.returncode


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
    args, extras = parse_args()
    verify_paths([args.program])
    board_path = convert_fen_to_board(Path(args.fen_board))
    cmd = build_command(args, board_path)
    append_extras(cmd, extras)
    exit_code = run_clingo(cmd)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
