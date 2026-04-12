#!/usr/bin/env python3
"""Compatibility helpers that wrap the new base-fragment workflow."""
from __future__ import annotations

from .base_programs import (
    BaseSpec,
    base_spec_facts_text,
    base_spec_to_cnl,
    load_base_specs,
)

BoardSpec = BaseSpec


def collect_board_specs(inputs: list[str]) -> list[BoardSpec]:
    return load_base_specs(inputs)


def board_spec_facts_text(spec: BoardSpec) -> str:
    return base_spec_facts_text(spec)


def board_spec_to_cnl(spec: BoardSpec) -> str:
    return base_spec_to_cnl(spec)
