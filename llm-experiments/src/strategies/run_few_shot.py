"""Few-shot prompting helpers (facts + CNL summary → ASP)."""
from __future__ import annotations

from pathlib import Path

from utils.board_utils import BoardSpec, board_spec_facts_text, board_spec_to_cnl
from utils.llm_utils import TokenUsage, llm_chat
from strategies.base import StrategyResult, run_board_strategy

DEFAULT_PROMPT = Path("prompts/few_shot/prompt.txt")


def generate_few_shot_program(
    *,
    client,
    model: str,
    prompt_template: str,
    base_program: str,
    board_description: str | None = None,
    max_new_tokens: int,
    temperature: float,
    usage: TokenUsage,
) -> str:
    prompt_body = prompt_template.rstrip()
    if board_description:
        prompt_body += "\n\nBoard description (CNL summary):\n" + board_description.strip() + "\n"
    prompt_body += "\nBase ASP fragment (copy verbatim before extending):\n" + base_program.strip() + "\n"
    prompt_body += (
        "\nExtend the fragment so the full program:\n"
        "  • selects exactly one legal white move using the provided legal_move/5 atoms;\n"
        "  • derives new_placement/3 for the board after that move;\n"
        "  • computes which squares white attacks after moving;\n"
        "  • checks if the black king is threatened;\n"
        "  • rules out escapes by analysing every black king move as well as blocks or captures;\n"
        "  • keeps only moves that yield checkmate and outputs move/5, in_check/1, can_escape/1, checkmate/1."
        "\nOutput the entire ASP program (base fragment + reasoning) with no Markdown."
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
    run_semantic: bool = False,
) -> None:

    def _generate(spec: BoardSpec, usage: TokenUsage, base_program: str) -> StrategyResult:
        board_description = board_spec_to_cnl(spec)
        asp_code = generate_few_shot_program(
            client=client,
            model=model,
            prompt_template=prompt_template,
            base_program=base_program,
            board_description=board_description,
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
        strategy_name="few_shot",
        subdir_name="few_shot",
        client=client,
        model=model,
        generate_fn=_generate,
        run_semantic=run_semantic,
    )
