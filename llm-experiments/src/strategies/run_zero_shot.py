"""Zero-shot strategy helpers (FEN → ASP in one hop)."""
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
    syntax_guardrail: str | None = None,
) -> None:

    def _generate(spec: BoardSpec, usage: TokenUsage, board_facts: str) -> StrategyResult:  # noqa: ARG001
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
