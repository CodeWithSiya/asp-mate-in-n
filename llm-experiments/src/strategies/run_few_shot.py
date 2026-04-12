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
    syntax_guardrail: str | None = None,
) -> None:

    def _generate(spec: BoardSpec, usage: TokenUsage, board_facts: str) -> StrategyResult:
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
    )
