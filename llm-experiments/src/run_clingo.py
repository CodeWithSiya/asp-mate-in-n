#!/usr/bin/env python3
"""Thin wrapper to run Clingo on generated ASP programs plus board facts."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Run Clingo against a generated ASP program.")
    parser.add_argument(
        "program",
        type=Path,
        help="Path to the ASP program to evaluate (e.g., results/<model>/<label>/generated.lp).",
    )
    parser.add_argument(
        "--board",
        action="append",
        default=[],
        dest="board_files",
        help="Board fact file(s) generated via fen_to_board_lp.py.",
    )
    parser.add_argument(
        "--facts",
        action="append",
        default=[],
        dest="extra_facts",
        help="Additional .lp fact files (e.g., constraints or gold labels).",
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


def build_command(args: argparse.Namespace) -> list[str]:
    cmd = [args.clingo_path, str(args.program)]
    cmd.extend(verify_paths(args.board_files))
    cmd.extend(verify_paths(args.extra_facts))
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


def main() -> None:
    args, extras = parse_args()
    verify_paths([args.program])
    cmd = build_command(args)
    append_extras(cmd, extras)
    exit_code = run_clingo(cmd)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
