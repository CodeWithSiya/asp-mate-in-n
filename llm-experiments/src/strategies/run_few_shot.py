#!/usr/bin/env python3
"""Generate mate-in-one ASP programs via few-shot prompting."""
from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path
import sys

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from utils.board_utils import (
    BoardSpec,
    collect_board_specs,
    board_spec_facts_text,
    board_spec_to_cnl,
)
from utils.clingo_utils import run_clingo_program, write_clingo_output
from utils.llm_utils import TokenUsage, build_anthropic_client, llm_chat
from utils.semantic_eval import semantic_validate, record_semantic_result
from utils.guardrails import load_syntax_guardrail

DEFAULT_PROMPT = Path("prompts/few_shot/prompt.txt")


def generate_few_shot_program(
    *,
    client,
    model: str,
    prompt_template: str,
    board_facts: str,
    board_description: str | None = None,
    max_new_tokens: int,
    temperature: float,
    usage: TokenUsage,
    syntax_guardrail: str | None = None,
) -> str:
    prompt_body = prompt_template.rstrip()
    if board_description:
        prompt_body += "\n\nBoard description (CNL summary):\n" + board_description.strip() + "\n"
    prompt_body += "\nBoard facts (ASP input):\n" + board_facts.strip() + "\n"
    prompt_body += (
        "\nGenerate only the ASP program that determines if there is a mate-in-one for"
        " this exact position. You may hard-code the provided coordinates and"
        " pieces; do not attempt to write a fully generic chess engine."
    )
    if syntax_guardrail:
        prompt_body += (
            "\n\nSyntax Guardrail (Clingo constraints):\n"
            f"{syntax_guardrail.strip()}\n"
        )
    system_prompt = "You are an expert in Answer Set Programming and chess."
    return llm_chat(
        client,
        model=model,
        stage="few_shot",
        system=system_prompt,
        user=prompt_body,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        token_usage=usage,
    )


def run_few_shot_on_boards(
    specs: list[BoardSpec],
    *,
    output_root: Path,
    client,
    model: str,
    prompt_template: str,
    prompt_file: Path,
    max_new_tokens: int,
    temperature: float,
    run_clingo: bool,
    syntax_guardrail: str | None = None,
) -> None:
    for spec in specs:
        usage = TokenUsage()
        board_id = spec.board_id
        board_facts = board_spec_facts_text(spec)
        board_description = board_spec_to_cnl(spec)
        asp_code = generate_few_shot_program(
            client=client,
            model=model,
            prompt_template=prompt_template,
            board_facts=board_facts,
            board_description=board_description,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            usage=usage,
            syntax_guardrail=syntax_guardrail,
        )

        board_dir = output_root / board_id / "few_shot"
        board_dir.mkdir(parents=True, exist_ok=True)
        (board_dir / "asp.lp").write_text(asp_code)

        metadata = {
            "id": board_id,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "variant": "few_shot",
            "model": model,
            "prompt_file": str(prompt_file),
            "fen_file": str(spec.fen_path),
            "expected_mate": spec.expected_mate,
            "token_usage": {
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "by_call": usage.by_call,
            },
        }

        if run_clingo:
            result = run_clingo_program(asp_code, board_facts)
            write_clingo_output(result, board_dir / "clingo_models.txt")
            if not result["parsed"]:
                status = "parse_error"
            elif result["total_models"] == 0:
                status = "no_models"
            else:
                status = "ok"
            metadata["clingo"] = {
                "status": status,
                "total_models": result["total_models"],
                "error": result["error_msg"],
                "ground_time": result["ground_time"],
                "solve_time": result["solve_time"],
                "total_time": result["total_time"],
                "statistics": result["statistics"],
                "stdout": result["stdout"],
                "stderr": result["stderr"],
                "logs": result.get("logs", []),
            }
        else:
            metadata["clingo"] = None

        sem_feedback, sem_score, sem_conf, semantic_json = semantic_validate(
            client,
            model=model,
            asp_code=asp_code,
            spec_text=board_facts,
            samples=3,
            temperature=0.1,
        )
        metadata["semantic"] = semantic_json
        record_semantic_result(
            strategy="few_shot",
            board_id=board_id,
            model=model,
            feedback=sem_feedback,
            score=sem_score,
            confidence=sem_conf,
            semantic=semantic_json,
        )

        (board_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        print(
            f"[{board_id}] wrote {board_dir}/asp.lp "
            f"(tokens {usage.prompt_tokens}+{usage.completion_tokens}) from {spec.fen_path}"
        )
        print(f"[{board_id}] semantic: {sem_feedback}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate ASP programs via few-shot prompting.")
    parser.add_argument("inputs", nargs="+", help=".fen board files or directories containing them.")
    parser.add_argument(
        "--prompt-file",
        type=Path,
        default=DEFAULT_PROMPT,
        help="Few-shot prompt template (default: prompts/few_shot/prompt.txt).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Root folder for per-board outputs (default: outputs/).",
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
        help="Max tokens per completion (default: 20000).",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.2, help="Sampling temperature (default: 0.2)."
    )
    parser.add_argument(
        "--clingo",
        action="store_true",
        help="Run Clingo immediately after generation and store outputs next to asp.lp.",
    )
    parser.add_argument(
        "--syntax-guardrail",
        action="store_true",
        help="Append the shared Clingo syntax guardrail to the few-shot prompt.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prompt_template = args.prompt_file.read_text().strip()
    specs = collect_board_specs(args.inputs)
    if not specs:
        raise SystemExit("No board files found.")

    client, model = build_anthropic_client(model_id=args.model_id)
    guardrail_text = load_syntax_guardrail() if args.syntax_guardrail else None
    print(f"Few-shot generation on {len(specs)} board(s) with model {model}")
    run_few_shot_on_boards(
        specs,
        output_root=args.output_dir,
        client=client,
        model=model,
        prompt_template=prompt_template,
        prompt_file=args.prompt_file,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        run_clingo=args.clingo,
        syntax_guardrail=guardrail_text,
    )


if __name__ == "__main__":
    main()
