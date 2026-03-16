#!/usr/bin/env python3
"""Convert Forsyth–Edwards Notation (FEN) strings to ASP board facts."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, List


FILES = "abcdefgh"
PIECE_NAME = {
    "p": "pawn",
    "n": "knight",
    "b": "bishop",
    "r": "rook",
    "q": "queen",
    "k": "king",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Translate a FEN string into ASP facts.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--fen", help="Raw FEN string.")
    source.add_argument("--fen-file", help="Path to a .fen file (first non-comment line is used).")
    parser.add_argument(
        "--output",
        type=Path,
        help="Destination .lp file. If omitted, the facts are printed to stdout.",
    )
    parser.add_argument(
        "--king-in-check",
        choices=["white", "black"],
        help="Emit king_in_check(Color). for the specified color.",
    )
    parser.add_argument(
        "--files-as",
        choices=["letters", "numbers"],
        default="numbers",
        help="Emit file coordinates as letters (a-h) or numbers (1-8).",
    )
    return parser.parse_args()


def load_fen_from_file(path: str | Path) -> str:
    data = Path(path).read_text().splitlines()
    for line in data:
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            return stripped
    raise ValueError(f"No FEN string found in {path}")


def fen_to_facts(fen: str, king_in_check: str | None = None, files_as: str = "numbers") -> List[str]:
    fen = fen.strip()
    parts = fen.split()
    if len(parts) < 2:
        raise ValueError("FEN must have at least piece placement and side-to-move fields.")
    placement, to_move = parts[0], parts[1]

    ranks = placement.split("/")
    if len(ranks) != 8:
        raise ValueError("Piece placement must contain 8 ranks.")

    facts: List[str] = []

    for rank_index, rank_text in enumerate(ranks):
        rank_number = 8 - rank_index
        file_index = 0
        for char in rank_text:
            if char.isdigit():
                file_index += int(char)
                continue
            color = "white" if char.isupper() else "black"
            piece_letter = char.lower()
            if piece_letter not in PIECE_NAME:
                raise ValueError(f"Unsupported piece letter '{char}' in FEN.")
            if file_index >= len(FILES):
                raise ValueError("File index out of range while parsing FEN.")
            file_value: str | int = FILES[file_index]
            if files_as == "numbers":
                file_value = file_index + 1
            piece_type = PIECE_NAME[piece_letter]
            facts.append(f"piece({color}, {piece_type}, {file_value}, {rank_number}).")
            if piece_type == "king":
                facts.append(f"king({color}, {file_value}, {rank_number}).")
            file_index += 1

    mover = {"w": "white", "b": "black"}.get(to_move.lower())
    if not mover:
        raise ValueError("Side-to-move field must be 'w' or 'b'.")
    facts.append(f"to_move({mover}).")

    if king_in_check:
        facts.append(f"king_in_check({king_in_check}).")

    return facts


def write_output(lines: Iterable[str], destination: Path | None) -> None:
    text = "\n".join(lines) + "\n"
    if destination:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text)
        print(f"Wrote {destination.resolve()}")
    else:
        sys.stdout.write(text)


def main() -> None:
    args = parse_args()
    fen = args.fen or load_fen_from_file(args.fen_file)
    facts = fen_to_facts(fen, king_in_check=args.king_in_check, files_as=args.files_as)
    write_output(facts, args.output)


if __name__ == "__main__":
    main()
