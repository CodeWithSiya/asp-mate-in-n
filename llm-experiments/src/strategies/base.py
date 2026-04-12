"""Shared helpers for per-board strategy runners."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable
import datetime
import json

from utils.board_utils import BoardSpec, board_spec_facts_text
from utils.clingo_utils import run_clingo_program, write_clingo_output
from utils.llm_utils import TokenUsage
from utils.reference_encodings import load_reference_program
from utils.semantic_eval import semantic_validate, record_semantic_result


@dataclass
class StrategyResult:
    """Output produced by a strategy-specific generator."""

    asp_code: str
    artifacts: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


GenerateFn = Callable[[BoardSpec, TokenUsage, str], StrategyResult]


def run_board_strategy(
    *,
    specs: Iterable[BoardSpec],
    output_root: Path,
    strategy_name: str,
    subdir_name: str,
    client,
    model: str,
    generate_fn: GenerateFn,
) -> None:
    """Generic driver that handles persistence, Clingo, and semantic scoring."""

    for spec in specs:
        usage = TokenUsage()
        board_id = spec.board_id
        board_dir = output_root / board_id / subdir_name
        board_dir.mkdir(parents=True, exist_ok=True)

        board_facts_text = board_spec_facts_text(spec)
        reference_code, reference_path = load_reference_program(board_id)

        result = generate_fn(spec, usage, board_facts_text)

        asp_path = board_dir / "asp.lp"
        asp_path.write_text(result.asp_code)

        for name, content in result.artifacts.items():
            (board_dir / name).write_text(content)

        clingo_result = run_clingo_program(result.asp_code, board_facts_text)
        write_clingo_output(clingo_result, board_dir / "clingo_models.txt")
        if not clingo_result["parsed"]:
            status = "parse_error"
        elif clingo_result["total_models"] == 0:
            status = "no_models"
        else:
            status = "ok"

        sem_feedback, sem_score, sem_conf, semantic_json = semantic_validate(
            client,
            model=model,
            asp_code=result.asp_code,
            spec_text=board_facts_text,
            samples=3,
            temperature=0.1,
            reference_code=reference_code,
            reference_path=reference_path,
            candidate_clingo=clingo_result,
        )

        metadata: dict[str, Any] = {
            "id": board_id,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "variant": strategy_name,
            "model": model,
            "fen_file": str(spec.fen_path),
            "expected_mate": spec.expected_mate,
            "token_usage": {
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "by_call": usage.by_call,
            },
            "clingo": {
                "status": status,
                "total_models": clingo_result["total_models"],
                "error": clingo_result["error_msg"],
                "ground_time": clingo_result["ground_time"],
                "solve_time": clingo_result["solve_time"],
                "total_time": clingo_result["total_time"],
                "statistics": clingo_result["statistics"],
                "stdout": clingo_result["stdout"],
                "stderr": clingo_result["stderr"],
                "logs": clingo_result.get("logs", []),
            },
            "semantic": semantic_json,
        }
        if reference_path:
            metadata["reference_encoding"] = str(reference_path)
        metadata.update(result.metadata)

        record_semantic_result(
            strategy=strategy_name,
            board_id=board_id,
            model=model,
            feedback=sem_feedback,
            score=sem_score,
            confidence=sem_conf,
            semantic=semantic_json,
        )

        (board_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

        print(
            f"[{board_id}] {strategy_name}: status={status} semantic={sem_score:.2f} "
            f"tokens {usage.prompt_tokens}+{usage.completion_tokens}"
        )
