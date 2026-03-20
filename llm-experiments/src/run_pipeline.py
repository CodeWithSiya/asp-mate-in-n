#!/usr/bin/env python3
"""
Chess Mate-in-1 LLM Pipeline. 
Adapted from LLMASP Honours Project By Keegan O'Brien, 2025.

Stages: Board Facts → CNL → CoT → ASP → Clingo
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

from fen_to_board_lp import load_fen_from_file, fen_to_facts

# TODO: (Potentially) move chain of thought to own file if not too complex. Might be better to keep here and let it be executable through the prompt
# TODO: Create Bash Script for running each experiment (Zero Shot, Few Shot, CoT Only, Pipeline)

# ── Token Usage (Reset Per Board) ────────────────────────────────────────────

USAGE: dict[str, Any] = {"prompt_tokens": 0, "completion_tokens": 0, "by_call": []}


def _reset_usage() -> None:
    USAGE["prompt_tokens"] = 0
    USAGE["completion_tokens"] = 0
    USAGE["by_call"].clear()


def _accumulate_usage(stage: str, model: str, usage: Any) -> None:
    if not usage:
        return
    pt = int(getattr(usage, "input_tokens", 0) or 0)
    ct = int(getattr(usage, "output_tokens", 0) or 0)
    USAGE["prompt_tokens"] += pt
    USAGE["completion_tokens"] += ct
    USAGE["by_call"].append({
        "stage": stage, "model": model,
        "prompt_tokens": pt, "completion_tokens": ct,
    })


# ── Anthropic client ──────────────────────────────────────────────────────────

def _build_anthropic_client():
    """Build the Anthropic client."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Export your Claude API key before using the anthropic backend."
        )
    try:
        from anthropic import Anthropic  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "anthropic Python package is required. Install with `pip install anthropic`."
        ) from exc
    return Anthropic(api_key=api_key)


def llm_chat(client, stage: str, model: str, system: str, user: str,
             max_new_tokens: int = 8000, temperature: float = 0.2) -> str:
    """System + user call to Claude. Returns text and accumulates token usage."""
    response = client.messages.create(
        model=model,
        max_tokens=max_new_tokens,
        temperature=temperature,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    _accumulate_usage(stage, model, getattr(response, "usage", None))
    parts = [
        block.text
        for block in response.content
        if getattr(block, "type", None) == "text"
    ]
    return "\n".join(parts).strip()


# ── Stage 1: Board facts → CNL ───────────────────────────────────────────────

def lp_to_cnl(lp_path: Path) -> str:
    """Reads a .lp board fact file and converts it to structured English."""
    fen_path = lp_path.with_suffix(".fen")
    fen = load_fen_from_file(fen_path)
    facts = fen_to_facts(fen)

    pieces: dict[str, list[str]] = {"white": [], "black": []}
    mover = "unknown"

    # Parse piece(Color, Type, X, Y), king(Color, X, Y) and to_move(Color) lines.
    for fact in facts:
        m = re.match(r"piece\((\w+),\s*(\w+),\s*(\d+),\s*(\d+)\)", fact.rstrip("."))
        if m:
            color, ptype, x, y = m.groups()
            pieces[color].append(f"{ptype} on ({x}, {y})")
            continue

        m = re.match(r"king\((\w+),\s*(\d+),\s*(\d+)\)", fact.rstrip("."))
        if m:
            color, x, y = m.groups()
            pieces[color].append(f"king on ({x}, {y})")
            continue

        m = re.match(r"to_move\((\w+)\)", fact.rstrip("."))
        if m:
            mover = m.group(1)

    lines = []
    for color in ("white", "black"):
        if pieces[color]:
            lines.append(f"{color.capitalize()} pieces: {', '.join(pieces[color])}.")
    lines.append(f"Side to move: {mover}.")
    lines.append("Board coordinates: files and ranks 1..8 (a=1, b=2, ..., h=8).")
    lines.append("Input predicates available: piece(Color, Type, X, Y).  king(Color, X, Y).  to_move(Color).")
    lines.append("Task: find a mate-in-one for the side to move.")
    return "\n".join(lines)


# ── Stage 2: CNL → CoT ───────────────────────────────────────────────────────

def generate_cot(client, model: str, cot_instruction: str, cnl_text: str,
                 max_new_tokens: int, temperature: float) -> str:
    print("  Calling Claude for CoT reasoning trace...")
    return llm_chat(
        client, stage="CNL->CoT", model=model,
        system=cot_instruction,
        user=f"Board description:\n{cnl_text}",
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )


# ── Stage 3: ASP generation ───────────────────────────────────────────────────

def generate_asp(client, model: str, asp_instruction: str,
                 cnl_text: str, cot_text: str,
                 max_new_tokens: int, temperature: float) -> str:
    user_content = f"Board description (CNL):\n{cnl_text}"
    if cot_text:
        user_content += f"\n\nReasoning trace (CoT):\n{cot_text}"
    user_content += "\n\nNow generate only the ASP program."

    print("  Calling Claude for ASP generation...")
    raw = llm_chat(
        client, stage="ASP-Gen", model=model,
        system=asp_instruction,
        user=user_content,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )
    return _strip_fences(raw)


def generate_asp_zero_shot(client, model: str, system: str, fen_text: str,
                            max_new_tokens: int, temperature: float) -> str:
    print("  Calling Claude for zero-shot ASP generation...")
    raw = llm_chat(
        client, stage="ASP-Gen-ZeroShot", model=model,
        system=system,
        user=f"FEN: {fen_text}\n\nGenerate only the ASP program.",
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )
    return _strip_fences(raw)

    
def _strip_fences(text: str) -> str:
    """Remove markdown code fences and leading line numbers (e.g. '001: ')."""
    lines = [l for l in text.splitlines() if not l.strip().startswith("```")]
    lines = [re.sub(r"^\s*\d+:\s*", "", l) for l in lines]
    return "\n".join(lines).strip()

# ── Stage 4: Clingo ───────────────────────────────────────────────────────────

def run_clingo_program(asp_code: str, board_facts: str) -> dict[str, Any]:
    """
    Concatenates generated ASP with board facts and solves via Clingo Python API.
    """
    combined = board_facts.strip() + "\n\n" + asp_code.strip()
    start = time.monotonic()

    try:
        import clingo
        ctl = clingo.Control()
        ctl.add("base", [], combined)
        ctl.ground([("base", [])])
        models: list[list] = []
        solve_start = time.monotonic()
        ctl.solve(on_model=lambda m: models.append(list(m.symbols(shown=True))))
        solve_end = time.monotonic()
        total = time.monotonic() - start
        return {
            "models": models,
            "total_models": len(models),
            "parsed": True,
            "error_msg": "",
            "ground_time": round(solve_start - start, 4),
            "solve_time": round(solve_end - solve_start, 4),
            "total_time": round(total, 4),
        }
    except Exception as exc:
        total = time.monotonic() - start
        return {
            "models": [],
            "total_models": 0,
            "parsed": False,
            "error_msg": str(exc),
            "ground_time": 0.0,
            "solve_time": 0.0,
            "total_time": round(total, 4),
        }
    
# ── Stage 5: Semantic Validation ──────────────────────────────────────────────

def _strip_md_fences(s: str) -> str:
    """Remove Markdown code fences from an LLM JSON response."""
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    return s.strip()

def _parse_json_response(raw: str) -> dict | None:
    """Try three increasingly lenient parses of an LLM JSON response."""
    for candidate in [raw, _strip_md_fences(raw)]:
        try:
            return json.loads(candidate)
        except Exception:
            pass
    m = re.search(r"\{[\s\S]*\}", _strip_md_fences(raw))
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    return None

def semantic_validate(client, model: str, asp_code: str, spec_text: str,
                      samples: int = 3, temperature: float = 0.1,
                      ) -> tuple[str, float, float, dict]:
    """
    LLM-based semantic validation of generated ASP against the board spec.
    
    Returns: (feedback_str, avg_score, avg_conf, semantic_json)
    """
    system_prompt = (
        "You are an expert in Answer Set Programming and chess.\n"
        "Given a board specification and a generated ASP program, check whether the ASP:\n"
        "  1. Correctly encodes legal piece movement for the relevant pieces.\n"
        "  2. Correctly detects when the opponent king is in check after the move.\n"
        "  3. Correctly determines the opponent has no legal escape (mate condition).\n"
        "  4. Uses the correct input predicates: piece(Color, Type, X, Y), "
        "king(Color, X, Y), to_move(Color).\n"
        "  5. Outputs: mate_move(Color, Type, FromX, FromY, ToX, ToY).\n"
        "Respond with STRICT JSON ONLY (no prose), schema:\n"
        '{ "verdict": "VALID|INVALID", "score": 0..1, "confidence": 0..1, "reasons": "string" }\n'
        "Scoring: 1.0 = all conditions correctly encoded; 0.7-0.9 = minor omissions; "
        "0.4-0.6 = partial; <0.4 = poor."
    )
    user_prompt = (
        f"Board specification:\n{spec_text}\n\n"
        f"Generated ASP program:\n{asp_code}\n\nReturn JSON only."
    )

    votes, scores, confidences, reasons_agg = [], [], [], []
    parsed_count = 0

    for i in range(max(1, samples)):
        raw = llm_chat(
            client, stage=f"Semantic-Validate/sample-{i + 1}", model=model,
            system=system_prompt, user=user_prompt, temperature=temperature,
        )
        data = _parse_json_response(raw)

        if isinstance(data, dict):
            verdict = str(data.get("verdict", "")).upper()
            if verdict not in {"VALID", "INVALID"}:
                verdict = "INVALID"
            try:
                score = max(0.0, min(1.0, float(data.get("score", 0.0))))
            except Exception:
                score = 0.0
            try:
                conf = max(0.0, min(1.0, float(data.get("confidence", 0.0))))
            except Exception:
                conf = 0.0
            votes.append(verdict)
            scores.append(score)
            confidences.append(conf)
            if data.get("reasons"):
                reasons_agg.append(str(data["reasons"]).strip())
            parsed_count += 1
        else:
            votes.append("UNPARSEABLE")
            cleaned = _strip_md_fences(raw)
            reasons_agg.append(cleaned[:400] + " …" if len(cleaned) > 400 else cleaned)

    avg_score = sum(scores) / parsed_count if parsed_count else 0.0
    avg_conf  = sum(confidences) / parsed_count if parsed_count else 0.0
    feedback  = f"Verdicts: {votes} | avg_score={avg_score:.3f} | avg_conf={avg_conf:.2f}"
    semantic_json = {
        "average_score": avg_score,
        "average_confidence": avg_conf,
        "verdicts": votes,
        "reasons": [r for r in reasons_agg if r],
    }
    return feedback, avg_score, avg_conf, semantic_json


# ── Board file helpers ────────────────────────────────────────────────────────

def collect_board_files(inputs: list[str]) -> list[Path]:
    """Accepts .lp files and/or directories. Directories scanned for *.lp files."""
    result: list[Path] = []
    for inp in inputs:
        p = Path(inp)
        if p.is_dir():
            result.extend(sorted(p.glob("*.lp")))
        elif p.suffix == ".lp" and p.exists():
            result.append(p)
        else:
            print(f"Warning: skipping '{inp}' (not a .lp file or directory)")
    return result


def read_expected_mate(lp_path: Path) -> str:
    """Read the '# Expected mates:' comment from the paired .fen file."""
    fen_path = lp_path.with_suffix(".fen")
    if not fen_path.exists():
        return ""
    for line in fen_path.read_text().splitlines():
        if "Expected" in line:
            return line.lstrip("#").strip()
    return ""

# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Chess mate-in-1 pipeline: board facts → CNL → CoT → ASP → Clingo."
    )
    parser.add_argument(
        "inputs", nargs="+",
        help=".lp board file(s) or director(ies) containing .lp files.",
    )
    parser.add_argument(
        "--variant",
        choices=["zero_shot", "cnl_only", "cnl_cot", "cot_only"],
        default="cnl_cot",
        help="Pipeline variant (default: cnl_cot).",
    )
    parser.add_argument(
        "--model-id",
        default=os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-20250514"),
        help="Anthropic Claude model name (env CLAUDE_MODEL).",
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=8000,
        help="Max tokens per LLM call (default: 8000).",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.2,
        help="Sampling temperature (default: 0.2).",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs"),
        help="Root directory for per-board output folders (default: outputs/).",
    )
    parser.add_argument(
        "--prompt-version", choices=["1", "2", "3"], default="1",
        help="Zero-shot prompt version (1/2/3) from prompts/zero-shot/ (only used with --variant zero_shot).",
    )
    args = parser.parse_args()

    prompts_dir = Path(__file__).parent.parent / "prompts"
    cot_instruction  = (prompts_dir / "chain-of-thought" / "chess-cot.txt").read_text()
    asp_instruction  = (prompts_dir / "chain-of-thought" / "chess-asp.txt").read_text()
    zero_shot_system = (prompts_dir / "zero-shot" / f"version{args.prompt_version}.txt").read_text()

    client = _build_anthropic_client()
    boards = collect_board_files(args.inputs)
    if not boards:
        print("No board files found. Exiting.")
        sys.exit(1)

    print(f"variant={args.variant!r}  model={args.model_id}  boards={len(boards)}")

    for lp_path in boards:
        board_id = lp_path.stem
        out_dir  = args.output_dir / board_id
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[{board_id}]")
        _reset_usage()

        board_facts = lp_path.read_text()
        expected    = read_expected_mate(lp_path)
        cnl_text = cot_text = asp_code = ""
        asp_gen_time = 0.0
        semantic_json: dict[str, Any] = {}
        sem_score = sem_conf = 0.0

        if args.variant == "cot_only":
            print("  Stage 1: Board Facts → CNL")
            cnl_text = lp_to_cnl(lp_path)
            (out_dir / "cnl.txt").write_text(cnl_text)

            print("  Stage 2: CNL → CoT")
            cot_text = generate_cot(
                client, model=args.model_id,
                cot_instruction=cot_instruction,
                cnl_text=cnl_text,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
            )
            (out_dir / "cot.txt").write_text(cot_text)

            metadata = {
                "id": board_id,
                "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
                "variant": args.variant,
                "prompt_version": None,
                "model": args.model_id,
                "expected_mate": expected,
                "token_usage": {
                    "prompt_tokens":     USAGE["prompt_tokens"],
                    "completion_tokens": USAGE["completion_tokens"],
                    "by_call":           USAGE["by_call"],
                },
            }
            (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
            print(f"  → CoT written  tokens={USAGE['prompt_tokens']}+{USAGE['completion_tokens']}")
            continue

        asp_gen_start = time.monotonic()
        if args.variant == "zero_shot":
            fen_text = load_fen_from_file(lp_path.with_suffix(".fen"))
            asp_code = generate_asp_zero_shot(
                client, model=args.model_id, system=zero_shot_system,
                fen_text=fen_text,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
            )
        else:
            print("  Stage 1: Board Facts → CNL")
            cnl_text = lp_to_cnl(lp_path)
            (out_dir / "cnl.txt").write_text(cnl_text)

            if args.variant == "cnl_cot":
                print("  Stage 2: CNL → CoT")
                cot_text = generate_cot(
                    client, model=args.model_id,
                    cot_instruction=cot_instruction,
                    cnl_text=cnl_text,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                )
                (out_dir / "cot.txt").write_text(cot_text)

            print("  Stage 3: CNL → ASP")
            asp_code = generate_asp(
                client, model=args.model_id,
                asp_instruction=asp_instruction,
                cnl_text=cnl_text,
                cot_text=cot_text,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
            )
        asp_gen_time = round(time.monotonic() - asp_gen_start, 4)

        (out_dir / "asp.lp").write_text(asp_code)

        print("  Stage 4: Clingo solve")
        result = run_clingo_program(asp_code, board_facts)

        if result["models"]:
            model_lines = []
            for i, model in enumerate(result["models"]):
                model_lines.append(f"Model {i + 1}:")
                for sym in model:
                    model_lines.append(f"  {sym}")
            (out_dir / "clingo_models.txt").write_text("\n".join(model_lines))
        else:
            (out_dir / "clingo_models.txt").write_text(
                "No models found.\n" + (result["error_msg"] or "")
            )

        if not result["parsed"]:
            status = "parse_error"
        elif result["total_models"] == 0:
            status = "no_models"
        else:
            status = "ok"

        print("  Stage 5: Semantic validation")
        spec_text = cnl_text if cnl_text else lp_path.with_suffix(".fen").read_text().splitlines()[0]
        sem_start = time.monotonic()
        sem_feedback, sem_score, sem_conf, semantic_json = semantic_validate(
            client, model=args.model_id,
            asp_code=asp_code,
            spec_text=spec_text,
            samples=3,
            temperature=0.1,
        )
        sem_time = round(time.monotonic() - sem_start, 4)
        print(f"  → {sem_feedback}")

        metadata = {
            "id": board_id,
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "variant": args.variant,
            "prompt_version": args.prompt_version if args.variant == "zero_shot" else None,
            "model": args.model_id,
            "expected_mate": expected,
            "status": status,
            "has_model": result["total_models"] > 0,
            "total_models": result["total_models"],
            "clingo_error": result["error_msg"],
            "semantic": semantic_json,
            "timings": {
                "asp_generation_time": asp_gen_time,
                "ground_time":         result["ground_time"],
                "solve_time":          result["solve_time"],
                "total_clingo_time":   result["total_time"],
                "semantic_validation_time": sem_time,
            },
            "token_usage": {
                "prompt_tokens":     USAGE["prompt_tokens"],
                "completion_tokens": USAGE["completion_tokens"],
                "by_call":           USAGE["by_call"],
            },
        }
        (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

        print(f"  → status={status!r}  models={result['total_models']}  "
              f"semantic={sem_score:.2f}  tokens={USAGE['prompt_tokens']}+{USAGE['completion_tokens']}")
        if expected:
            print(f"  → expected: {expected}")

    print(f"\nDone. Results in {args.output_dir}/")

if __name__ == "__main__":
    main()