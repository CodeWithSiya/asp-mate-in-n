#!/usr/bin/env python3
"""Shared Anthropic helper functions for strategy runners and pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import os
import re


try:  # Optional import so modules can be loaded without Anthropic installed.
    from anthropic import Anthropic  # type: ignore
except Exception:  # pragma: no cover - we only fail at runtime when needed
    Anthropic = None  # type: ignore


@dataclass
class TokenUsage:
    """Tracks prompt/completion token usage per strategy invocation."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    by_call: list[dict[str, Any]] = field(default_factory=list)

    def reset(self) -> None:
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.by_call.clear()

    def accumulate(self, stage: str, model: str, usage: Any) -> None:
        if not usage:
            return
        pt = int(getattr(usage, "input_tokens", 0) or 0)
        ct = int(getattr(usage, "output_tokens", 0) or 0)
        self.prompt_tokens += pt
        self.completion_tokens += ct
        self.by_call.append(
            {
                "stage": stage,
                "model": model,
                "prompt_tokens": pt,
                "completion_tokens": ct,
            }
        )


def build_anthropic_client(model_id: str | None = None):
    """Create an Anthropic client and resolve the model name."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Export your Claude API key before using the anthropic backend."
        )
    if Anthropic is None:
        raise RuntimeError(
            "anthropic Python package is required. Install with `pip install anthropic`."
        )
    resolved_model = model_id or os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-6")
    client = Anthropic(api_key=api_key)
    return client, resolved_model


def extract_text_blocks(response: Any) -> str:
    parts: list[str] = []
    content: Iterable[Any] = getattr(response, "content", [])
    for block in content:
        block_type = getattr(block, "type", None)
        if block_type == "text":
            parts.append(block.text)
    if not parts and hasattr(response, "text"):
        parts.append(response.text)
    return "\n".join(parts).strip()


def llm_chat(
    client: Any,
    *,
    model: str,
    stage: str,
    system: str | None,
    user: str,
    max_new_tokens: int,
    temperature: float,
    token_usage: TokenUsage | None = None,
) -> str:
    response = client.messages.create(
        model=model,
        system=system,
        max_tokens=max_new_tokens,
        temperature=temperature,
        messages=[{"role": "user", "content": user}],
    )
    if token_usage is not None:
        token_usage.accumulate(stage, model, getattr(response, "usage", None))
    return strip_code_fences(extract_text_blocks(response))


def strip_code_fences(text: str) -> str:
    """Remove Markdown code fences and leading line numbers."""
    if not text:
        return ""
    lines = [line for line in text.splitlines() if not line.strip().startswith("```")]
    cleaned = [re.sub(r"^\s*\d+:\s*", "", line) for line in lines]
    return "\n".join(cleaned).strip()
