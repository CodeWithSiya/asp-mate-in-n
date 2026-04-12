#!/usr/bin/env python3
"""Run Clingo on a standalone ASP program (already includes base facts)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from utils.clingo_utils import run_clingo_program, write_clingo_output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Clingo against a generated ASP program.")
    parser.add_argument(
        "program",
        type=Path,
        help="Path to the ASP program to evaluate (should already contain the base fragment).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path for the solver transcript (default: alongside the ASP program).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.program.exists():
        raise FileNotFoundError(f"Missing ASP program: {args.program}")
    asp_code = args.program.read_text()
    result = run_clingo_program(asp_code)
    output_path = args.output or args.program.with_name("clingo_manual_run.txt")
    write_clingo_output(result, output_path)
    print(f"Wrote solver output to {output_path}")
    status = 0 if result.get("parsed", True) else 1
    sys.exit(status)


if __name__ == "__main__":
    main()
