#!/usr/bin/env python3
"""Standalone chain-of-thought runner (CNL -> CoT -> ASP)."""
from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path
import sys

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from utils.board_utils import BoardSpec, collect_board_specs, board_spec_to_cnl, board_spec_facts_text
from utils.llm_utils import TokenUsage, build_anthropic_client, llm_chat
from utils.clingo_utils import run_clingo_program, write_clingo_output
from utils.semantic_eval import semantic_validate, record_semantic_result
from utils.guardrails import load_syntax_guardrail

COT_PROMPT = Path("prompts/chain-of-thought/chess-cot.txt")
ASP_PROMPT = Path("prompts/chain-of-thought/chess-asp.txt")


def generate_cot_trace(
    *,
    client,
    model: str,
    instruction: str,
    cnl_text: str,
    max_new_tokens: int,
    temperature: float,
    usage: TokenUsage | None = None,
) -> str:
    return llm_chat(
        client,
        model=model,
        stage="cnl_to_cot",
        system=instruction,
        user=f"Board description:\n{cnl_text}",
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        token_usage=usage,
    )


def generate_cot_asp(
    *,
    client,
    model: str,
    instruction: str,
    cnl_text: str,
    cot_text: str,
    max_new_tokens: int,
    temperature: float,
    usage: TokenUsage | None = None,
    feedback: str = "",
    previous_asp: str | None = None,
    syntax_guardrail: str | None = None,
) -> str:
    user_content = f"Board description (CNL):\n{cnl_text}"
    if cot_text:
        user_content += f"\n\nReasoning trace (CoT):\n{cot_text}"
    user_content += "\n\nNow generate only the ASP program."
    if syntax_guardrail:
        user_content += (
            "\n\nSyntax Guardrail (Clingo constraints):\n"
            f"{syntax_guardrail.strip()}\n"
        )
    if feedback.strip():
        user_content += (
            "\n\nPrevious solver feedback:\n"
            f"{feedback.strip()}\n"
            "Revise the ASP to address these issues."
        )
        if syntax_guardrail:
            user_content += (
                "\n\nSyntax Guardrail Reminder:\n"
                f"{syntax_guardrail.strip()}\n"
            )
    if previous_asp:
        user_content += (
            "\n\nPrevious ASP attempt:\n```asp\n"
            f"{previous_asp.strip()}\n"
            "```"
        )
    return llm_chat(
        client,
        model=model,
        stage="cot_to_asp",
        system=instruction,
        user=user_content,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        token_usage=usage,
    )


def run_chain_of_thought_on_boards(
    specs: list[BoardSpec],
    *,
    output_root: Path,
    client,
    model: str,
    cot_instruction: str,
    asp_instruction: str,
    cot_prompt_path: Path,
    asp_prompt_path: Path,
    max_new_tokens: int,
    temperature: float,
    run_clingo: bool,
    syntax_guardrail: str | None = None,
) -> None:
    for spec in specs:
        usage = TokenUsage()
        board_id = spec.board_id
        board_facts_text = board_spec_facts_text(spec)
        cnl_text = board_spec_to_cnl(spec)
        cot_text = generate_cot_trace(
            client=client,
            model=model,
            instruction=cot_instruction,
            cnl_text=cnl_text,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            usage=usage,
        )
        asp_code = generate_cot_asp(
            client=client,
            model=model,
            instruction=asp_instruction,
            cnl_text=cnl_text,
            cot_text=cot_text,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            usage=usage,
            syntax_guardrail=syntax_guardrail,
        )

        board_dir = output_root / board_id / "chain_of_thought"
        board_dir.mkdir(parents=True, exist_ok=True)
        (board_dir / "cnl.txt").write_text(cnl_text)
        (board_dir / "cot.txt").write_text(cot_text)
        (board_dir / "asp.lp").write_text(asp_code)

        metadata = {
            "id": board_id,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "variant": "chain_of_thought",
            "model": model,
            "cot_prompt": str(cot_prompt_path),
            "asp_prompt": str(asp_prompt_path),
            "fen_file": str(spec.fen_path),
            "expected_mate": spec.expected_mate,
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
            strategy="chain_of_thought",
            board_id=board_id,
            model=model,
            feedback=sem_feedback,
            score=sem_score,
            confidence=sem_conf,
            semantic=semantic_json,
        )

        print(f"[{board_id}] semantic: {sem_feedback}")
        (board_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        print(
            f"[{board_id}] wrote chain-of-thought artefacts (tokens {usage.prompt_tokens}+{usage.completion_tokens})"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate ASP programs via CNL -> CoT -> ASP prompting.")
    parser.add_argument("inputs", nargs="+", help=".fen board files or directories containing them.")
    parser.add_argument(
        "--cot-prompt",
        type=Path,
        default=COT_PROMPT,
        help="Path to the chain-of-thought reasoning instruction (default: prompts/chain-of-thought/chess-cot.txt).",
    )
    parser.add_argument(
        "--asp-prompt",
        type=Path,
        default=ASP_PROMPT,
        help="Path to the ASP generation instruction (default: prompts/chain-of-thought/chess-asp.txt).",
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
        help="Append the shared Clingo syntax guardrail to the ASP-generation prompt.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cot_instruction = args.cot_prompt.read_text()
    asp_instruction = args.asp_prompt.read_text()
    specs = collect_board_specs(args.inputs)
    if not specs:
        raise SystemExit("No board files found.")

    client, model = build_anthropic_client(model_id=args.model_id)
    guardrail_text = load_syntax_guardrail() if args.syntax_guardrail else None
    print(f"Chain-of-thought generation on {len(specs)} board(s) with model {model}")
    run_chain_of_thought_on_boards(
        specs,
        output_root=args.output_dir,
        client=client,
        model=model,
        cot_instruction=cot_instruction,
        asp_instruction=asp_instruction,
        cot_prompt_path=args.cot_prompt,
        asp_prompt_path=args.asp_prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        run_clingo=args.clingo,
        syntax_guardrail=guardrail_text,
    )


if __name__ == "__main__":
    main()
