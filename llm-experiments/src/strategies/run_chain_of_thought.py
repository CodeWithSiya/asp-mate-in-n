"""Chain-of-thought helpers (structured context → reasoning → ASP)."""
from __future__ import annotations

from pathlib import Path

from strategies.base import StrategyResult, run_board_strategy
from utils.board_utils import BoardSpec
from utils.llm_utils import TokenUsage, llm_chat

COT_PROMPT = Path("prompts/chain-of-thought/chess-cot.txt")
ASP_PROMPT = Path("prompts/chain-of-thought/chess-asp.txt")


def generate_cot_trace(
    *,
    client,
    model: str,
    instruction: str,
    context_label: str,
    context_text: str,
    max_new_tokens: int,
    temperature: float,
    usage: TokenUsage | None = None,
) -> str:
    return llm_chat(
        client,
        model=model,
        stage="cnl_to_cot",
        system=instruction,
        user=f"{context_label}:\n{context_text}",
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        token_usage=usage,
    )


def generate_cot_asp(
    *,
    client,
    model: str,
    instruction: str,
    context_label: str,
    context_text: str,
    cot_text: str,
    max_new_tokens: int,
    temperature: float,
    usage: TokenUsage | None = None,
    feedback: str = "",
    previous_asp: str | None = None,
    syntax_guardrail: str | None = None,
) -> str:
    user_content = f"{context_label}:\n{context_text}"
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
    syntax_guardrail: str | None = None,
) -> None:

    def _generate(spec: BoardSpec, usage: TokenUsage, board_facts: str) -> StrategyResult:
        context_label = "Board facts (ASP input)"
        cot_text = generate_cot_trace(
            client=client,
            model=model,
            instruction=cot_instruction,
            context_label=context_label,
            context_text=board_facts,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            usage=usage,
        )
        asp_code = generate_cot_asp(
            client=client,
            model=model,
            instruction=asp_instruction,
            context_label=context_label,
            context_text=board_facts,
            cot_text=cot_text,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            usage=usage,
            syntax_guardrail=syntax_guardrail,
        )
        return StrategyResult(
            asp_code=asp_code,
            artifacts={"cot.txt": cot_text},
            metadata={
                "cot_prompt": str(cot_prompt_path),
                "asp_prompt": str(asp_prompt_path),
            },
        )

    run_board_strategy(
        specs=specs,
        output_root=output_root,
        strategy_name="chain_of_thought",
        subdir_name="chain_of_thought",
        client=client,
        model=model,
        generate_fn=_generate,
    )
