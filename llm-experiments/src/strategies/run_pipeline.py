#!/usr/bin/env python3
"""Chess Mate-in-1 LLM pipeline orchestration."""
from __future__ import annotations

import argparse
import datetime
import json
import sys
import time
import shutil
from pathlib import Path
from typing import Any

# Ensure the repo's src/ root is on sys.path when running as a script.
SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from utils.board_utils import (
    board_spec_facts_text,
    board_spec_to_cnl,
    collect_board_specs,
)
from utils.llm_utils import TokenUsage, build_anthropic_client
from strategies.run_chain_of_thought import generate_cot_trace, generate_cot_asp
from strategies.run_zero_shot import generate_zero_shot_asp
from utils.clingo_utils import run_clingo_program, write_clingo_output
from utils.semantic_eval import semantic_validate, record_semantic_result
from utils.guardrails import load_syntax_guardrail



def _clingo_status(result: dict[str, Any]) -> str:
    if not result.get("parsed", True):
        return "parse_error"
    if result.get("total_models", 0) == 0:
        return "no_models"
    return "ok"


def _get_error_text(result: dict[str, Any]) -> str:
    stderr = (result.get("stderr") or "").strip()
    if stderr:
        return stderr
    logs = result.get("logs") or []
    if logs:
        return "\n".join(logs)
    return (result.get("error_msg") or "").strip()


def _build_feedback_message(result: dict[str, Any], attempt: int) -> str:
    lines: list[str] = [
        f"Attempt {attempt} failed when running Clingo.",
    ]
    if not result.get("parsed", True):
        err = _get_error_text(result)
        if err:
            snippet = err[:1200]
            lines.append("Clingo reported a parsing error:")
            lines.append(snippet)
        else:
            lines.append("Clingo reported a parsing error (no diagnostic output was captured).")
    elif result.get("total_models", 0) == 0:
        lines.append(
            "Clingo ran successfully but found zero stable models. "
            "Please adjust the ASP so it derives at least one mate_move/6 atom."
        )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chess mate-in-1 pipeline: board facts → CNL → CoT → ASP → Clingo."
    )
    parser.add_argument("inputs", nargs="+", help=".fen board files or directories containing them.")
    parser.add_argument(
        "--variant",
        choices=["zero_shot", "cnl_only", "cnl_cot", "cot_only"],
        default="cnl_cot",
        help="Pipeline variant (default: cnl_cot).",
    )
    parser.add_argument(
        "--zero-shot-prompt",
        type=Path,
        default=Path("prompts/zero-shot/prompt.txt"),
        help="Zero-shot system prompt template (default: prompts/zero-shot/prompt.txt).",
    )
    parser.add_argument(
        "--cot-prompt",
        type=Path,
        default=Path("prompts/chain-of-thought/chess-cot.txt"),
        help="CNL→CoT instruction file (default: prompts/chain-of-thought/chess-cot.txt).",
    )
    parser.add_argument(
        "--asp-prompt",
        type=Path,
        default=Path("prompts/chain-of-thought/chess-asp.txt"),
        help="CoT→ASP instruction file (default: prompts/chain-of-thought/chess-asp.txt).",
    )
    parser.add_argument(
        "--model-id",
        default=None,
        help="Anthropic Claude model (default: $CLAUDE_MODEL or claude-sonnet-4-6).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=20000,
        help="Max tokens per LLM call (default: 20000).",
    )
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature (default: 0.2).")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Root directory for per-board output folders (default: outputs/).",
    )
    parser.add_argument(
        "--clingo-retries",
        type=int,
        default=2,
        help="Number of additional times to regenerate the ASP when Clingo fails (default: 2).",
    )
    parser.add_argument(
        "--syntax-guardrail",
        action="store_true",
        help="Append the shared Clingo syntax guardrail to ASP-generation prompts.",
    )
    return parser.parse_args()


def run_pipeline_with_args(args: argparse.Namespace) -> None:
    guardrail_text = getattr(args, "syntax_guardrail_text", None)
    if guardrail_text is None and getattr(args, "syntax_guardrail", False):
        guardrail_text = load_syntax_guardrail()
    zero_shot_system = args.zero_shot_prompt.read_text()
    cot_instruction = args.cot_prompt.read_text()
    asp_instruction = args.asp_prompt.read_text()

    specs = collect_board_specs(args.inputs)
    if not specs:
        print("No board files found. Exiting.")
        sys.exit(1)

    client, model_name = build_anthropic_client(model_id=args.model_id)
    print(f"variant={args.variant!r}  model={model_name}  boards={len(specs)}")

    for spec in specs:
        board_id = spec.board_id
        out_dir = args.output_dir / board_id / "pipeline"
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[{board_id}]")
        usage = TokenUsage()

        board_facts = board_spec_facts_text(spec)
        expected = spec.expected_mate
        cnl_text = ""
        cot_text = ""
        asp_code = ""
        semantic_json: dict[str, Any] = {}
        sem_score = sem_conf = 0.0
        attempt_logs: list[dict[str, Any]] = []

        if args.variant == "cot_only":
            print("  Stage 1: Board Facts → CNL")
            cnl_text = board_spec_to_cnl(spec)
            (out_dir / "cnl.txt").write_text(cnl_text)

            print("  Stage 2: CNL → CoT")
            cot_text = generate_cot_trace(
                client=client,
                model=model_name,
                instruction=cot_instruction,
                cnl_text=cnl_text,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                usage=usage,
            )
            (out_dir / "cot.txt").write_text(cot_text)

            metadata = {
                "id": board_id,
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "variant": args.variant,
                "model": model_name,
                "expected_mate": expected,
                "token_usage": {
                    "prompt_tokens": usage.prompt_tokens,
                    "completion_tokens": usage.completion_tokens,
                    "by_call": usage.by_call,
                },
            }
            (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
            print(
                f"  → CoT written  tokens={usage.prompt_tokens}+{usage.completion_tokens}"
            )
            continue

        if args.variant != "zero_shot":
            print("  Stage 1: Board Facts → CNL")
            cnl_text = board_spec_to_cnl(spec)
            (out_dir / "cnl.txt").write_text(cnl_text)

            if args.variant == "cnl_cot":
                print("  Stage 2: CNL → CoT")
                cot_text = generate_cot_trace(
                    client=client,
                    model=model_name,
                    instruction=cot_instruction,
                    cnl_text=cnl_text,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    usage=usage,
                )
                (out_dir / "cot.txt").write_text(cot_text)

        asp_gen_start = time.monotonic()
        max_attempts = max(1, args.clingo_retries + 1)
        feedback = ""
        last_result: dict[str, Any] | None = None
        final_asp_path = out_dir / "asp.lp"
        final_clingo_path = out_dir / "clingo_models.txt"
        prev_asp: str | None = None

        for attempt_idx in range(1, max_attempts + 1):
            if args.variant == "zero_shot":
                print(f"  Stage 3: FEN → ASP (attempt {attempt_idx}/{max_attempts})")
                asp_code = generate_zero_shot_asp(
                    client=client,
                    model=model_name,
                    system_prompt=zero_shot_system,
                    fen_text=spec.fen_text,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    usage=usage,
                    feedback=feedback,
                    previous_asp=prev_asp,
                    syntax_guardrail=guardrail_text,
                )
            else:
                print(f"  Stage 3: CNL → ASP (attempt {attempt_idx}/{max_attempts})")
                asp_code = generate_cot_asp(
                    client=client,
                    model=model_name,
                    instruction=asp_instruction,
                    cnl_text=cnl_text,
                    cot_text=cot_text,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    usage=usage,
                    feedback=feedback,
                    previous_asp=prev_asp,
                    syntax_guardrail=guardrail_text,
                )

            attempt_asp_path = out_dir / f"asp_attempt{attempt_idx}.lp"
            attempt_asp_path.write_text(asp_code)
            prev_asp = asp_code

            print(f"  Stage 4: Clingo solve (attempt {attempt_idx}/{max_attempts})")
            last_result = run_clingo_program(asp_code, board_facts)
            attempt_clingo_path = out_dir / f"clingo_models_attempt{attempt_idx}.txt"
            write_clingo_output(last_result, attempt_clingo_path)

            attempt_status = _clingo_status(last_result)
            attempt_logs.append(
                {
                    "attempt": attempt_idx,
                    "status": attempt_status,
                    "asp_file": str(attempt_asp_path),
                    "clingo_file": str(attempt_clingo_path),
                    "total_models": last_result["total_models"],
                    "error": last_result["error_msg"],
                    "ground_time": last_result["ground_time"],
                    "solve_time": last_result["solve_time"],
                    "total_time": last_result["total_time"],
                    "statistics": last_result["statistics"],
                    "stdout": last_result["stdout"],
                    "stderr": last_result["stderr"],
                    "logs": last_result.get("logs", []),
                }
            )

            if attempt_status == "ok" or attempt_idx == max_attempts:
                shutil.copyfile(attempt_asp_path, final_asp_path)
                shutil.copyfile(attempt_clingo_path, final_clingo_path)
                break

            feedback = _build_feedback_message(last_result, attempt_idx)
            if guardrail_text:
                feedback = (
                    f"{feedback}\n\nSyntax Guardrail (Clingo constraints):\n"
                    f"{guardrail_text.strip()}\n"
                )
            print(f"    ↻ retrying (status={attempt_status})")

        asp_gen_time = round(time.monotonic() - asp_gen_start, 4)

        if last_result is None:
            raise RuntimeError("Clingo never ran; this should not happen.")

        status = _clingo_status(last_result)

        print("  Stage 5: Semantic validation")
        spec_text = board_facts
        sem_start = time.monotonic()
        sem_feedback, sem_score, sem_conf, semantic_json = semantic_validate(
            client,
            model=model_name,
            asp_code=asp_code,
            spec_text=spec_text,
            samples=3,
            temperature=0.1,
        )
        sem_time = round(time.monotonic() - sem_start, 4)
        print(f"  → {sem_feedback}")

        record_semantic_result(
            strategy="pipeline",
            board_id=board_id,
            model=model_name,
            feedback=sem_feedback,
            score=sem_score,
            confidence=sem_conf,
            semantic=semantic_json,
        )

        metadata = {
            "id": board_id,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "variant": args.variant,
            "model": model_name,
            "zero_shot_prompt": str(args.zero_shot_prompt) if args.variant == "zero_shot" else None,
            "cot_prompt": str(args.cot_prompt) if args.variant in {"cnl_cot", "cot_only"} else None,
            "asp_prompt": str(args.asp_prompt) if args.variant in {"cnl_only", "cnl_cot"} else None,
            "fen_file": str(spec.fen_path),
            "expected_mate": expected,
            "status": status,
            "has_model": last_result["total_models"] > 0,
            "total_models": last_result["total_models"],
            "clingo_error": last_result["error_msg"],
            "clingo_statistics": last_result["statistics"],
            "clingo_stdout": last_result["stdout"],
            "clingo_stderr": last_result["stderr"],
            "clingo_logs": last_result.get("logs", []),
            "clingo_attempts": attempt_logs,
            "clingo_retries_allowed": args.clingo_retries,
            "semantic": semantic_json,
            "timings": {
                "asp_generation_time": asp_gen_time,
                "ground_time": last_result["ground_time"],
                "solve_time": last_result["solve_time"],
                "total_clingo_time": last_result["total_time"],
                "semantic_validation_time": sem_time,
            },
            "token_usage": {
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "by_call": usage.by_call,
            },
        }
        (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

        print(
            f"  → status={status!r}  models={last_result['total_models']}  "
            f"semantic={sem_score:.2f}  tokens={usage.prompt_tokens}+{usage.completion_tokens}"
        )
        if expected:
            print(f"  → expected: {expected}")

    print(f"\nDone. Results in {args.output_dir}/")


def main() -> None:
    args = parse_args()
    args.syntax_guardrail_text = load_syntax_guardrail() if args.syntax_guardrail else None
    run_pipeline_with_args(args)


if __name__ == "__main__":
    main()
