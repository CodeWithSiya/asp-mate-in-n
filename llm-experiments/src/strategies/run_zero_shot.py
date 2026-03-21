#!/usr/bin/env python3
"""Standalone zero-shot runner (LLM generates ASP directly from FEN)."""
from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path
import sys

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from utils.board_utils import BoardSpec, collect_board_specs, board_spec_facts_text
from utils.llm_utils import TokenUsage, build_anthropic_client, llm_chat
from utils.clingo_utils import run_clingo_program, write_clingo_output
from utils.semantic_eval import semantic_validate, record_semantic_result
from utils.guardrails import load_syntax_guardrail

DEFAULT_PROMPT = Path("prompts/zero-shot/prompt.txt")


def generate_zero_shot_asp(
    *,
    client,
    model: str,
    system_prompt: str,
    fen_text: str,
    max_new_tokens: int,
    temperature: float,
    usage: TokenUsage | None = None,
    feedback: str = "",
    previous_asp: str | None = None,
    syntax_guardrail: str | None = None,
) -> str:
    user_prompt = f"FEN: {fen_text}\n\nGenerate only the ASP program."
    if feedback.strip():
        user_prompt += (
            "\n\nPrevious solver feedback:\n"
            f"{feedback.strip()}\n"
            "Revise the ASP accordingly."
        )
    if previous_asp:
        user_prompt += (
            "\n\nPrevious ASP attempt:\n```asp\n"
            f"{previous_asp.strip()}\n"
            "```"
        )
    if syntax_guardrail:
        user_prompt += (
            "\n\nSyntax Guardrail (Clingo constraints):\n"
            f"{syntax_guardrail.strip()}\n"
        )
    return llm_chat(
        client,
        model=model,
        stage="zero_shot",
        system=system_prompt,
        user=user_prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        token_usage=usage,
    )


def run_zero_shot_on_boards(
    specs: list[BoardSpec],
    *,
    output_root: Path,
    client,
    model: str,
    system_prompt: str,
    max_new_tokens: int,
    temperature: float,
    prompt_file: Path,
    run_clingo: bool,
    syntax_guardrail: str | None = None,
) -> None:
    for spec in specs:
        board_id = spec.board_id
        usage = TokenUsage()
        asp_code = generate_zero_shot_asp(
            client=client,
            model=model,
            system_prompt=system_prompt,
            fen_text=spec.fen_text,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            usage=usage,
            syntax_guardrail=syntax_guardrail,
        )
        board_facts_text = board_spec_facts_text(spec)

        board_dir = output_root / board_id / "zero_shot"
        board_dir.mkdir(parents=True, exist_ok=True)
        (board_dir / "asp.lp").write_text(asp_code)

        metadata = {
            "id": board_id,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "variant": "zero_shot",
            "model": model,
            "prompt_file": str(prompt_file),
            "expected_mate": spec.expected_mate,
            "fen_file": str(spec.fen_path),
            "token_usage": {
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "by_call": usage.by_call,
            },
        }
        if run_clingo:
            result = run_clingo_program(asp_code, board_facts_text)
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
            spec_text=board_facts_text,
            samples=3,
            temperature=0.1,
        )
        metadata["semantic"] = semantic_json
        record_semantic_result(
            strategy="zero_shot",
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
    parser = argparse.ArgumentParser(description="Generate ASP programs via zero-shot prompting.")
    parser.add_argument("inputs", nargs="+", help=".fen board files or directories containing them.")
    parser.add_argument(
        "--prompt-file",
        type=Path,
        default=DEFAULT_PROMPT,
        help="Zero-shot system prompt template (default: prompts/zero-shot/prompt.txt).",
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
        help="Run Clingo immediately after generation and store results next to asp.lp.",
    )
    parser.add_argument(
        "--syntax-guardrail",
        action="store_true",
        help="Append the shared Clingo syntax guardrail to the zero-shot prompt.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    system_prompt = args.prompt_file.read_text()
    specs = collect_board_specs(args.inputs)
    if not specs:
        raise SystemExit("No usable board files found.")

    client, model = build_anthropic_client(model_id=args.model_id)
    guardrail_text = load_syntax_guardrail() if args.syntax_guardrail else None
    print(f"Zero-shot generation on {len(specs)} board(s) with model {model}")
    run_zero_shot_on_boards(
        specs,
        output_root=args.output_dir,
        client=client,
        model=model,
        system_prompt=system_prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        prompt_file=args.prompt_file,
        run_clingo=args.clingo,
        syntax_guardrail=guardrail_text,
    )


if __name__ == "__main__":
    main()
