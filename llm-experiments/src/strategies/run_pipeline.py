"""Chess Mate-in-1 LLM pipeline orchestration."""
from __future__ import annotations

import datetime
import json
import time
import shutil
from pathlib import Path
from typing import Any

from utils.board_utils import board_spec_facts_text, board_spec_to_cnl
from utils.llm_utils import TokenUsage
from strategies.run_chain_of_thought import generate_cot_trace, generate_cot_asp
from strategies.run_zero_shot import generate_zero_shot_asp
from utils.clingo_utils import run_clingo_program, write_clingo_output
from utils.semantic_eval import semantic_validate, record_semantic_result
from utils.reference_encodings import load_reference_program



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


def run_pipeline(
    specs,
    *,
    output_root: Path,
    client,
    model: str,
    zero_shot_system: str,
    cot_instruction: str,
    asp_instruction: str,
    zero_shot_prompt_path: Path,
    cot_prompt_path: Path,
    asp_prompt_path: Path,
    variant: str,
    max_new_tokens: int,
    temperature: float,
    clingo_retries: int,
) -> None:
    print(f"variant={variant!r}  model={model}  boards={len(specs)}")

    for spec in specs:
        board_id = spec.board_id
        out_dir = output_root / board_id / "pipeline"
        out_dir.mkdir(parents=True, exist_ok=True)
        reference_code, reference_path = load_reference_program(board_id)

        print(f"\n[{board_id}]")
        usage = TokenUsage()

        base_fragment = board_spec_facts_text(spec)
        (out_dir / "base.lp").write_text(base_fragment)
        expected = spec.expected_mate
        cnl_text = ""
        cot_text = ""
        asp_code = ""
        semantic_json: dict[str, Any] = {}
        sem_score = sem_conf = 0.0
        attempt_logs: list[dict[str, Any]] = []

        if variant == "cot_only":
            print("  Stage 1: Board Facts → CNL")
            cnl_text = board_spec_to_cnl(spec)
            (out_dir / "cnl.txt").write_text(cnl_text)

            print("  Stage 2: CNL → CoT")
            cot_text = generate_cot_trace(
                client=client,
                model=model,
                instruction=cot_instruction,
                context_label="Board description (CNL)",
                context_text=cnl_text,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                usage=usage,
            )
            (out_dir / "cot.txt").write_text(cot_text)

            metadata = {
                "id": board_id,
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "variant": variant,
                "model": model,
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

        if variant != "zero_shot":
            print("  Stage 1: Board Facts → CNL")
            cnl_text = board_spec_to_cnl(spec)
            (out_dir / "cnl.txt").write_text(cnl_text)

            if variant == "cnl_cot":
                print("  Stage 2: CNL → CoT")
                cot_text = generate_cot_trace(
                    client=client,
                    model=model,
                    instruction=cot_instruction,
                    context_label="Board description (CNL)",
                    context_text=cnl_text,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    usage=usage,
                )
                (out_dir / "cot.txt").write_text(cot_text)

        asp_gen_start = time.monotonic()
        max_attempts = max(1, clingo_retries + 1)
        feedback = ""
        last_result: dict[str, Any] | None = None
        final_asp_path = out_dir / "asp.lp"
        final_clingo_path = out_dir / "clingo_models.txt"
        prev_asp: str | None = None

        for attempt_idx in range(1, max_attempts + 1):
            if variant == "zero_shot":
                print(f"  Stage 3: Base fragment → ASP (attempt {attempt_idx}/{max_attempts})")
                asp_code = generate_zero_shot_asp(
                    client=client,
                    model=model,
                    system_prompt=zero_shot_system,
                    base_program=base_fragment,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    usage=usage,
                    feedback=feedback,
                    previous_asp=prev_asp,
                )
            else:
                print(f"  Stage 3: CNL → ASP (attempt {attempt_idx}/{max_attempts})")
                asp_code = generate_cot_asp(
                    client=client,
                    model=model,
                    instruction=asp_instruction,
                    context_label="Board description (CNL)",
                    context_text=cnl_text,
                    base_program=base_fragment,
                    cot_text=cot_text,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    usage=usage,
                    feedback=feedback,
                    previous_asp=prev_asp,
                )

            attempt_asp_path = out_dir / f"asp_attempt{attempt_idx}.lp"
            attempt_asp_path.write_text(asp_code)
            prev_asp = asp_code

            print(f"  Stage 4: Clingo solve (attempt {attempt_idx}/{max_attempts})")
            last_result = run_clingo_program(asp_code)
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
            print(f"    ↻ retrying (status={attempt_status})")

        asp_gen_time = round(time.monotonic() - asp_gen_start, 4)

        if last_result is None:
            raise RuntimeError("Clingo never ran; this should not happen.")

        status = _clingo_status(last_result)

        print("  Stage 5: Semantic validation")
        spec_text = base_fragment
        sem_start = time.monotonic()
        sem_feedback, sem_score, sem_conf, semantic_json = semantic_validate(
            client,
            model=model,
            asp_code=asp_code,
            spec_text=spec_text,
            samples=3,
            temperature=0.1,
            reference_code=reference_code,
            reference_path=reference_path,
            candidate_clingo=last_result,
        )
        sem_time = round(time.monotonic() - sem_start, 4)
        print(f"  → {sem_feedback}")

        record_semantic_result(
            strategy="pipeline",
            board_id=board_id,
            model=model,
            feedback=sem_feedback,
            score=sem_score,
            confidence=sem_conf,
            semantic=semantic_json,
        )

        metadata = {
            "id": board_id,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "variant": variant,
            "model": model,
            "zero_shot_prompt": str(zero_shot_prompt_path) if variant == "zero_shot" else None,
            "cot_prompt": str(cot_prompt_path) if variant in {"cnl_cot", "cot_only"} else None,
            "asp_prompt": str(asp_prompt_path) if variant in {"cnl_only", "cnl_cot"} else None,
            "base_file": str(spec.base_path),
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
            "clingo_retries_allowed": clingo_retries,
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
        if reference_path:
            metadata["reference_encoding"] = str(reference_path)
        (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

        print(
            f"  → status={status!r}  models={last_result['total_models']}  "
            f"semantic={sem_score:.2f}  tokens={usage.prompt_tokens}+{usage.completion_tokens}"
        )
        if expected:
            print(f"  → expected: {expected}")

    print(f"\nDone. Results in {output_root}/")
