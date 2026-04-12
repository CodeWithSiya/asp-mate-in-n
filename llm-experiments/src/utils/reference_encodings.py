#!/usr/bin/env python3
"""Lookup helpers for manual reference encodings used during semantic validation."""
from __future__ import annotations

from pathlib import Path

# Root directory that contains the curated manual encodings. By default we look for
# the sibling repository ../encoding/src relative to this project. Update this path
# if the manual files live somewhere else on your machine.
DEFAULT_REFERENCE_ROOT = (Path(__file__).resolve().parents[2] / ".." / "encoding" / "src").resolve()


# Hard-coded mapping from board identifiers (derived from FEN filenames) to the
# corresponding manual .lp file names. Extend this dict whenever new reference
# encodings become available.
BOARD_REFERENCE_MAP: dict[str, str] = {
    # Example mapping: "manual_easy": "mate-in-n-easy.lp",
    # "manual_medium": "mate-in-n-medium.lp",
}


def resolve_reference_path(board_id: str, root: Path | None = None) -> Path | None:
    """Return the manual encoding path for the given board id, if configured."""

    rel_name = BOARD_REFERENCE_MAP.get(board_id)
    if not rel_name:
        return None
    base = (root or DEFAULT_REFERENCE_ROOT)
    candidate = (base / rel_name).resolve()
    if not candidate.exists():
        print(f"[semantic] reference file '{candidate}' missing for board '{board_id}'")
        return None
    return candidate


def load_reference_program(board_id: str, root: Path | None = None) -> tuple[str | None, Path | None]:
    """Load the manual encoding text for the provided board id, if available."""

    path = resolve_reference_path(board_id, root=root)
    if not path:
        return None, None
    try:
        return path.read_text(), path
    except Exception as exc:  # noqa: BLE001 - best effort, fall back if unreadable
        print(f"[semantic] failed to read reference '{path}': {exc}")
        return None, None


__all__ = [
    "BOARD_REFERENCE_MAP",
    "DEFAULT_REFERENCE_ROOT",
    "load_reference_program",
    "resolve_reference_path",
]
