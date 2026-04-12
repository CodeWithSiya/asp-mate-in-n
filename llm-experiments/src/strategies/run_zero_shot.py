"""Zero-shot strategy helpers (base fragment → full ASP)."""
from __future__ import annotations

from pathlib import Path

from utils.board_utils import BoardSpec
from utils.llm_utils import TokenUsage, llm_chat
from strategies.base import StrategyResult, run_board_strategy

DEFAULT_PROMPT = Path("prompts/zero-shot/prompt.txt")


def generate_zero_shot_asp(
    *,
    client,
    model: str,
    system_prompt: str,
    base_program: str,
    max_new_tokens: int,
    temperature: float,
    usage: TokenUsage | None = None,
    feedback: str = "",
    previous_asp: str | None = None,
) -> str:
    header = (
        "You are given a curated mate-in-one base program fragment. It already"
        " encodes the board state (placement/5), shared movement helpers"
        " (queen_moves/4, king_moves/4, blocked/5, etc.) that derive legal_move/5,"
        " plus to_move/1. Copy it verbatim at the top of your answer, then extend"
        " it so the final ASP selects exactly one white move, derives the"
        " post-move board, detects threats/check, evaluates every escape (king"
        " move, block, capture), and rules out moves that fail to deliver"
        " checkmate."
    )
    base_section = "Base program fragment (copy exactly before extending):\n" + base_program.strip()
    checklist = (
        "Your completion must implement:\n"
        "1. Choice rule that selects exactly one legal white move.\n"
        "2. new_placement rules describing the board after that move.\n"
        "3. Threat detection predicates showing which squares white attacks.\n"
        "4. Check detection for the black king.\n"
        "5. Escape detection: opponent king moves, interpositions, captures.\n"
        "6. Checkmate constraint rejecting moves unless they trap the king.\n"
        "Output the FULL program (base fragment + reasoning)."
    )
    user_prompt = "\n\n".join([header, base_section, checklist])
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
) -> None:

    def _generate(spec: BoardSpec, usage: TokenUsage, board_facts: str) -> StrategyResult:  # noqa: ARG001
        asp_code = generate_zero_shot_asp(
            client=client,
            model=model,
            system_prompt=system_prompt,
            base_program=board_facts,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            usage=usage,
        )
        return StrategyResult(
            asp_code=asp_code,
            metadata={"prompt_file": str(prompt_file)},
        )

    run_board_strategy(
        specs=specs,
        output_root=output_root,
        strategy_name="zero_shot",
        subdir_name="zero_shot",
        client=client,
        model=model,
        generate_fn=_generate,
    )
