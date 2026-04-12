#!/usr/bin/env python3
"""Coordinator script that runs every prompting strategy sequentially."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from utils.board_utils import collect_board_specs
from utils.llm_utils import build_anthropic_client
from strategies.run_zero_shot import (
    DEFAULT_PROMPT as ZERO_SHOT_PROMPT,
    run_zero_shot_on_boards,
)
from strategies.run_few_shot import (
    DEFAULT_PROMPT as FEW_SHOT_PROMPT,
    run_few_shot_on_boards,
)
from strategies.run_chain_of_thought import (
    COT_PROMPT as DEFAULT_COT_PROMPT,
    ASP_PROMPT as DEFAULT_ASP_PROMPT,
    run_chain_of_thought_on_boards,
)
from strategies.run_pipeline import run_pipeline


DEFAULT_STRATEGIES: Sequence[str] = (
    "zero_shot",
    "few_shot",
    "chain_of_thought",
    "pipeline",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run every prompting strategy sequentially on the same set of base ASP fragments."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="*_base.lp files or directories containing them (e.g., data/base).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Root directory for per-board outputs (default: outputs/).",
    )
    parser.add_argument(
        "--model-id",
        default=None,
        help="Anthropic Claude model name (default: $CLAUDE_MODEL or claude-sonnet-4-6).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=20000,
        help="Maximum tokens per LLM call (default: 20000).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="SamplingTemperature shared across strategies (default: 0.2).",
    )
    parser.add_argument(
        "--zero-shot-prompt",
        type=Path,
        default=ZERO_SHOT_PROMPT,
        help=f"Zero-shot system prompt template (default: {ZERO_SHOT_PROMPT}).",
    )
    parser.add_argument(
        "--few-shot-prompt",
        type=Path,
        default=FEW_SHOT_PROMPT,
        help=f"Few-shot prompt template (default: {FEW_SHOT_PROMPT}).",
    )
    parser.add_argument(
        "--cot-prompt",
        type=Path,
        default=DEFAULT_COT_PROMPT,
        help=f"CNL→CoT instruction file (default: {DEFAULT_COT_PROMPT}).",
    )
    parser.add_argument(
        "--asp-prompt",
        type=Path,
        default=DEFAULT_ASP_PROMPT,
        help=f"CoT→ASP instruction file (default: {DEFAULT_ASP_PROMPT}).",
    )
    parser.add_argument(
        "--clingo-retries",
        type=int,
        default=2,
        help="Extra ASP regeneration attempts for the pipeline strategy (default: 2).",
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        choices=DEFAULT_STRATEGIES,
        default=list(DEFAULT_STRATEGIES),
        help="Subset of strategies to run (default: all).",
    )
    return parser.parse_args()


def run_experiment(args: argparse.Namespace) -> None:
    specs = collect_board_specs(args.inputs)
    if not specs:
        print("No board files found. Exiting.")
        sys.exit(1)

    client, model_name = build_anthropic_client(model_id=args.model_id)
    selected = list(dict.fromkeys(args.strategies))  # preserve order, drop duplicates
    spec_count = len(specs)
    print(
        f"Running strategies {selected} on {spec_count} board(s) "
        f"with model {model_name} (outputs → {args.output_dir})."
    )

    if "zero_shot" in selected:
        print("\n=== Zero-shot strategy ===")
        zero_shot_system = args.zero_shot_prompt.read_text()
        run_zero_shot_on_boards(
            specs,
            output_root=args.output_dir,
            client=client,
            model=model_name,
            system_prompt=zero_shot_system,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            prompt_file=args.zero_shot_prompt,
        )

    if "few_shot" in selected:
        print("\n=== Few-shot strategy ===")
        prompt_template = args.few_shot_prompt.read_text().strip()
        run_few_shot_on_boards(
            specs,
            output_root=args.output_dir,
            client=client,
            model=model_name,
            prompt_template=prompt_template,
            prompt_file=args.few_shot_prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )

    if "chain_of_thought" in selected:
        print("\n=== Chain-of-thought strategy ===")
        cot_instruction = args.cot_prompt.read_text()
        asp_instruction = args.asp_prompt.read_text()
        run_chain_of_thought_on_boards(
            specs,
            output_root=args.output_dir,
            client=client,
            model=model_name,
            cot_instruction=cot_instruction,
            asp_instruction=asp_instruction,
            cot_prompt_path=args.cot_prompt,
            asp_prompt_path=args.asp_prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )

    if "pipeline" in selected:
        print("\n=== Pipeline strategy ===")
        zero_shot_system = args.zero_shot_prompt.read_text()
        cot_instruction = args.cot_prompt.read_text()
        asp_instruction = args.asp_prompt.read_text()
        run_pipeline(
            specs,
            output_root=args.output_dir,
            client=client,
            model=model_name,
            zero_shot_system=zero_shot_system,
            cot_instruction=cot_instruction,
            asp_instruction=asp_instruction,
            zero_shot_prompt_path=args.zero_shot_prompt,
            cot_prompt_path=args.cot_prompt,
            asp_prompt_path=args.asp_prompt,
            variant="cnl_cot",
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            clingo_retries=args.clingo_retries,
        )

    print("\nAll requested strategies completed.")


def main() -> None:
    args = parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
