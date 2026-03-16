"""Convenience wrapper to run all few-shot prompt variants in one go."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from run_prompt import build_anthropic_generator, run_prompt_file

DEFAULT_PROMPTS = [
    Path("prompts/few_shot/version1.txt"),
    Path("prompts/few_shot/version2.txt"),
    Path("prompts/few_shot/version3.txt"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all few-shot prompts with Anthropic Claude.")
    parser.add_argument(
        "--prompt", action="append", help="Explicit prompt file(s). If omitted, version1-3 are used.",
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=10000, help="Max tokens per completion",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.2, help="Sampling temperature",
    )
    parser.add_argument(
        "--model-id",
        default=os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-20250514"),
        help="Anthropic Claude model name (env CLAUDE_MODEL).",
    )
    args = parser.parse_args()

    prompt_paths = [Path(p) for p in args.prompt] if args.prompt else DEFAULT_PROMPTS
    generate_fn, model_id = build_anthropic_generator(model_id=args.model_id)

    for prompt_path in prompt_paths:
        label = prompt_path.stem
        print(f"\n=== Running few-shot prompt: {prompt_path} ===")
        run_prompt_file(
            prompt_path,
            generate_fn,
            model_id,
            output_name=f"few_shot_{label}",
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )

if __name__ == "__main__":
    main()
