"""Regression tests for the FEN → ASP fact converter."""

from __future__ import annotations

from pathlib import Path
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from utils.fen_to_board_lp import fen_to_facts


def test_fen_to_facts_emits_numeric_coordinates_and_kings() -> None:
    fen = "8/8/8/8/8/8/8/K6k w - - 0 1"
    facts = fen_to_facts(fen)

    assert "piece(white, king, 1, 1)." in facts
    assert "king(white, 1, 1)." in facts
    assert "piece(black, king, 8, 1)." in facts
    assert "king(black, 8, 1)." in facts
    assert facts[-1] == "to_move(white)."


def test_fen_to_facts_supports_letters_and_in_check_flag() -> None:
    fen = "8/8/8/8/8/8/8/K6k b - - 0 1"
    facts = fen_to_facts(fen, king_in_check="white", files_as="letters")

    assert "piece(white, king, a, 1)." in facts
    assert "piece(black, king, h, 1)." in facts
    assert "king_in_check(white)." in facts
    assert facts[-1] == "king_in_check(white)."


@pytest.mark.parametrize(
    "fen",
    [
        "8/8/8/8/8/8/8 w - - 0 1",  # not enough ranks
        "8/8/8/8/8/8/8/K6x w - - 0 1",  # invalid piece letter
        "8/8/8/8/8/8/8/K6k x - - 0 1",  # invalid mover
    ],
)
def test_fen_to_facts_rejects_invalid_inputs(fen: str) -> None:
    with pytest.raises(ValueError):
        fen_to_facts(fen)
