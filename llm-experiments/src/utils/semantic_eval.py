#!/usr/bin/env python3
"""Shared helpers for LLM-based semantic validation of ASP outputs."""
from __future__ import annotations

import datetime
import json
import re
from pathlib import Path
from typing import Any

from .llm_utils import llm_chat
from functools import lru_cache

from .clingo_utils import run_clingo_program


def _strip_md_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _parse_json_response(raw: str) -> dict | None:
    for candidate in [raw, _strip_md_fences(raw)]:
        try:
            return json.loads(candidate)
        except Exception:  # noqa: BLE001 - best-effort parsing
            pass
    match = re.search(r"\{[\s\S]*\}", _strip_md_fences(raw))
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:  # noqa: BLE001
            pass
    return None


def _summarize_clingo_result(result: dict[str, Any] | None) -> dict[str, Any] | None:
    if result is None:
        return None
    summary: dict[str, Any] = {
        "status": "ok",
        "models": [],
        "parsed": bool(result.get("parsed", True)),
        "total_models": int(result.get("total_models", 0) or 0),
        "error": result.get("error_msg") or "",
    }
    if not summary["parsed"]:
        summary["status"] = "parse_error"
        return summary
    if summary["total_models"] == 0:
        summary["status"] = "no_models"
        return summary

    models = []
    for model in result.get("models", []) or []:
        atoms = sorted(str(symbol) for symbol in model)
        models.append(atoms)
    summary["models"] = sorted(models)
    return summary


def _compare_reference_to_candidate(
    reference_summary: dict[str, Any] | None,
    candidate_summary: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if reference_summary is None:
        return None

    status: str
    if reference_summary["status"] != "ok":
        status = f"reference_{reference_summary['status']}"
    elif candidate_summary is None:
        status = "candidate_unavailable"
    elif candidate_summary["status"] != "ok":
        status = f"candidate_{candidate_summary['status']}"
    else:
        status = "match" if reference_summary["models"] == candidate_summary["models"] else "mismatch"

    return {
        "status": status,
        "reference_models": reference_summary.get("models", []),
        "candidate_models": (candidate_summary or {}).get("models", []),
        "reference_total_models": reference_summary.get("total_models", 0),
        "candidate_total_models": (candidate_summary or {}).get("total_models", 0),
    }


def semantic_validate(
    client: Any,
    *,
    model: str,
    asp_code: str,
    spec_text: str,
    samples: int = 3,
    temperature: float = 0.1,
    max_new_tokens: int = 1500,
    reference_code: str | None = None,
    reference_path: Path | None = None,
    candidate_clingo: dict[str, Any] | None = None,
) -> tuple[str, float, float, dict]:
    """LLM-based semantic validation of generated ASP against a board specification."""

    system_prompt = (
        "You are an expert in Answer Set Programming and chess.\n"
        "Given a board specification and a generated ASP program, check whether the ASP:\n"
        "  1. Correctly encodes legal piece movement for the relevant pieces.\n"
        "  2. Correctly detects when the opponent king is in check after the move.\n"
        "  3. Correctly determines the opponent has no legal escape (mate condition).\n"
        "  4. Uses the correct input predicates: placement(Color, Piece, Id, Row, Col), to_move(Color).\n"
        "  5. Outputs: move(Id, R1, C1, R2, C2), in_check(Color), can_escape(Color), checkmate(Color).\n"
        "Respond with STRICT JSON ONLY (no prose), schema:\n"
        '{ "verdict": "VALID|INVALID", "score": 0..1, "confidence": 0..1, "reasons": "string" }\n'
        "Scoring: 1.0 = all conditions correctly encoded; 0.7-0.9 = minor omissions; 0.4-0.6 = partial; <0.4 = poor."
    )
    if reference_code:
        user_prompt = (
            f"Board specification:\n{spec_text}\n\n"
            f"Reference ASP (known correct):\n{reference_code}\n\n"
            f"Candidate ASP program:\n{asp_code}\n\nReturn JSON only."
        )
        system_prompt += (
            "\nCompare the candidate against the reference: award VALID only if the candidate"
            " satisfies the mate-in-one criteria even when it differs. Cite differences in your reasons."
        )
    else:
        user_prompt = (
            f"Board specification:\n{spec_text}\n\n"
            f"Generated ASP program:\n{asp_code}\n\nReturn JSON only."
        )

    votes: list[str] = []
    scores: list[float] = []
    confidences: list[float] = []
    reasons_agg: list[str] = []
    parsed_count = 0

    for idx in range(max(1, samples)):
        raw = llm_chat(
            client,
            model=model,
            stage=f"Semantic-Validate/sample-{idx + 1}",
            system=system_prompt,
            user=user_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        data = _parse_json_response(raw)
        if isinstance(data, dict):
            verdict = str(data.get("verdict", "")).upper()
            if verdict not in {"VALID", "INVALID"}:
                verdict = "INVALID"
            try:
                score = max(0.0, min(1.0, float(data.get("score", 0.0))))
            except Exception:  # noqa: BLE001
                score = 0.0
            try:
                conf = max(0.0, min(1.0, float(data.get("confidence", 0.0))))
            except Exception:  # noqa: BLE001
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
    avg_conf = sum(confidences) / parsed_count if parsed_count else 0.0
    feedback = f"Verdicts: {votes} | avg_score={avg_score:.3f} | avg_conf={avg_conf:.2f}"
    semantic_json = {
        "average_score": avg_score,
        "average_confidence": avg_conf,
        "verdicts": votes,
        "reasons": [reason for reason in reasons_agg if reason],
    }

    reference_summary = None
    candidate_summary = None
    if reference_code:
        reference_summary = _summarize_clingo_result(_solve_reference(reference_code))
    if candidate_clingo is not None:
        candidate_summary = _summarize_clingo_result(candidate_clingo)
    elif reference_code:
        candidate_summary = _summarize_clingo_result(run_clingo_program(asp_code))

    reference_comparison = _compare_reference_to_candidate(reference_summary, candidate_summary)
    if reference_comparison or reference_path:
        semantic_json["reference"] = {
            "path": str(reference_path) if reference_path else None,
            "comparison": reference_comparison,
        }
    return feedback, avg_score, avg_conf, semantic_json


@lru_cache(maxsize=128)
def _solve_reference(reference_code: str) -> dict[str, Any]:
    return run_clingo_program(reference_code)


def record_semantic_result(
    *,
    strategy: str,
    board_id: str,
    model: str,
    feedback: str,
    score: float,
    confidence: float,
    semantic: dict,
    results_root: Path = Path("results"),
) -> Path:
    """Persist semantic evaluation data and refresh the Markdown summary table."""

    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    payload = {
        "board": board_id,
        "strategy": strategy,
        "model": model,
        "timestamp": timestamp,
        "average_score": score,
        "average_confidence": confidence,
        "feedback": feedback,
        "semantic": semantic,
    }

    semantic_root = results_root / "semantic"
    strategy_dir = semantic_root / strategy
    strategy_dir.mkdir(parents=True, exist_ok=True)
    output_path = strategy_dir / f"{board_id}.json"
    output_path.write_text(json.dumps(payload, indent=2) + "\n")

    _write_summary_table(semantic_root)
    return output_path


def _write_summary_table(semantic_root: Path) -> None:
    records: list[dict[str, Any]] = []
    if not semantic_root.exists():
        semantic_root.mkdir(parents=True, exist_ok=True)
    for json_path in semantic_root.rglob("*.json"):
        try:
            data = json.loads(json_path.read_text())
        except Exception:  # noqa: BLE001
            continue
        data["_source_path"] = str(json_path)
        records.append(data)

    records.sort(key=lambda rec: rec.get("timestamp", ""))

    lines = ["# Semantic Evaluation Summary", ""]
    if not records:
        lines.append("_No semantic evaluations recorded yet._")
    else:
        lines.append("| Board | Strategy | Model | Score | Confidence | Verdicts | Timestamp |")
        lines.append("| --- | --- | --- | ---: | ---: | --- | --- |")
        for rec in records:
            verdicts = ", ".join(rec.get("semantic", {}).get("verdicts", [])) or "—"
            lines.append(
                "| {board} | {strategy} | {model} | {score:.3f} | {conf:.2f} | {verdicts} | {timestamp} |".format(
                    board=rec.get("board", "?"),
                    strategy=rec.get("strategy", "?"),
                    model=rec.get("model", "?"),
                    score=float(rec.get("average_score", 0.0)),
                    conf=float(rec.get("average_confidence", 0.0)),
                    verdicts=verdicts,
                    timestamp=rec.get("timestamp", ""),
                )
            )

    summary_path = semantic_root / "summary.md"
    summary_path.write_text("\n".join(lines) + "\n")
