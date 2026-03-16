"""Run Anthropic Claude prompts (zero- or few-shot) with minimal configuration."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Callable


DEFAULT_ZERO_SHOT_PROMPT = Path("prompts/zero_shot_template.txt")
DEFAULT_FEW_SHOT_PROMPTS = (
    Path("prompts/few_shot/version1.txt"),
    Path("prompts/few_shot/version2.txt"),
    Path("prompts/few_shot/version3.txt"),
)


def _resolve_prompt_paths(prompt_type: str, prompt_files: list[str] | None) -> list[Path]:
    if prompt_files:
        return [Path(p) for p in prompt_files]
    if prompt_type == "few-shot":
        return list(DEFAULT_FEW_SHOT_PROMPTS)
    return [DEFAULT_ZERO_SHOT_PROMPT]


def main():
    parser = argparse.ArgumentParser(description="Run Anthropic Claude on predefined prompt templates.")
    parser.add_argument(
        "--prompt-type",
        choices=["zero-shot", "few-shot"],
        default="zero-shot",
        help="Choose between the default zero-shot template or all few-shot variants.",
    )
    parser.add_argument(
        "--prompt-file",
        action="append",
        dest="prompt_files",
        help="Override prompt template(s). If omitted, defaults depend on --prompt-type.",
    )
    parser.add_argument(
        "--model-id",
        default=os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-20250514"),
        help="Anthropic Claude model name (env CLAUDE_MODEL).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=10000,
        help="Maximum tokens to sample for each completion.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature.",
    )
    args = parser.parse_args()

    prompt_paths = _resolve_prompt_paths(args.prompt_type, args.prompt_files)
    generate_fn, model_id = build_anthropic_generator(model_id=args.model_id)

    multiple = len(prompt_paths) > 1
    for prompt_path in prompt_paths:
        if not args.prompt_files and args.prompt_type == "zero-shot":
            label = "zero_shot"
        else:
            label = prompt_path.stem
            if multiple:
                label = f"{args.prompt_type.replace('-', '_')}_{label}"
        run_prompt_file(
            prompt_path,
            generate_fn,
            model_id,
            output_name=label,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )


def run_prompt_file(
    prompt_path: Path,
    generate: Callable[[str, int, float], str],
    model_id: str,
    output_name: str | None = None,
    max_new_tokens: int = 10000,
    temperature: float = 0.2,
    output_filename: str = "generated.lp",
):
    prompt = prompt_path.read_text().strip()
    print(f"Using model: {model_id}")
    print(f"Loaded prompt from {prompt_path} ({len(prompt.split())} words)")

    completion = generate(prompt, max_new_tokens, temperature)
    completion = completion.strip()

    label = output_name or prompt_path.stem
    output_dir = Path("results") / model_id.replace("/", "_") / label
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / output_filename
    output_file.write_text(completion)
    print(f"Saved completion to {output_file}")

def build_anthropic_generator(model_id: str | None = None) -> tuple[Callable[[str, int, float], str], str]:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Export your Claude API key before using the anthropic backend."
        )
    try:
        from anthropic import Anthropic  # type: ignore
    except ImportError as exc:  # pragma: no cover - dependency check
        raise RuntimeError("anthropic Python package is required. Install with `pip install anthropic`.") from exc

    model_name = model_id or os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-20250514")
    client = Anthropic(api_key=api_key)

    def _generate(prompt: str, max_new_tokens: int, temperature: float) -> str:
        response = client.messages.create(
            model=model_name,
            max_tokens=max_new_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        parts: list[str] = []
        for block in response.content:
            block_type = getattr(block, "type", None)
            if block_type == "text":
                parts.append(block.text)
        if not parts and hasattr(response, "text"):
            parts.append(response.text)
        completion = "\n".join(parts).strip()
        return completion

    return _generate, model_name


if __name__ == "__main__":
    main()
