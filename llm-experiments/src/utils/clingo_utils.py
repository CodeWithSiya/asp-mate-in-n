#!/usr/bin/env python3
"""Shared helpers for running Clingo on generated programs."""
from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import time
import io
from contextlib import redirect_stdout, redirect_stderr


def _convert_stats(stats: Any) -> Any:
    try:
        keys = list(stats.keys())
    except Exception:
        pass
    else:
        return {key: _convert_stats(stats[key]) for key in keys}

    try:
        length = len(stats)
    except Exception:
        pass
    else:
        if not isinstance(stats, (str, bytes)):
            try:
                return [_convert_stats(stats[i]) for i in range(length)]
            except Exception:
                pass

    try:
        return float(stats)
    except Exception:
        return stats


def run_clingo_program(asp_code: str, board_facts: str) -> dict[str, Any]:
    """Concatenate ASP with board facts and solve via Clingo's Python API."""
    combined = board_facts.strip() + "\n\n" + asp_code.strip()
    start = time.monotonic()
    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()

    logs: list[str] = []

    with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
        try:
            import clingo

            def _logger(code: str, message: str) -> None:
                logs.append(f"{code}: {message}")

            ctl = clingo.Control(logger=_logger)
            ctl.add("base", [], combined)
            ctl.ground([("base", [])])
            models: list[list] = []
            solve_start = time.monotonic()
            ctl.solve(on_model=lambda m: models.append(list(m.symbols(shown=True))))
            solve_end = time.monotonic()
            total = time.monotonic() - start
            stats = None
            try:
                stats = _convert_stats(ctl.statistics)
            except Exception:
                stats = None
            result = {
                "models": models,
                "total_models": len(models),
                "parsed": True,
                "error_msg": "",
                "ground_time": round(solve_start - start, 4),
                "solve_time": round(solve_end - solve_start, 4),
                "total_time": round(total, 4),
                "statistics": stats,
            }
        except Exception as exc:  # noqa: BLE001
            total = time.monotonic() - start
            result = {
                "models": [],
                "total_models": 0,
                "parsed": False,
                "error_msg": str(exc),
                "ground_time": 0.0,
                "solve_time": 0.0,
                "total_time": round(total, 4),
                "statistics": None,
            }

    result["stdout"] = stdout_buf.getvalue()
    result["stderr"] = stderr_buf.getvalue()
    result["logs"] = logs
    return result


def write_clingo_output(result: dict[str, Any], destination: Path) -> None:
    """Write human-readable model output plus statistics to disk."""
    lines: list[str] = []
    if result.get("models"):
        for i, model in enumerate(result["models"]):
            lines.append(f"Model {i + 1}:")
            for sym in model:
                lines.append(f"  {sym}")
    else:
        lines.append("No models found.")
        if result.get("error_msg"):
            lines.append(result["error_msg"])

    if result.get("statistics") is not None:
        stats_json = json.dumps(result["statistics"], indent=2, sort_keys=True)
        lines.append("")
        lines.append("Clingo statistics (--stats equivalent):")
        lines.append(stats_json)

    stdout_text = (result.get("stdout") or "").strip()
    if stdout_text:
        lines.append("")
        lines.append("Clingo stdout:")
        lines.append(stdout_text)

    stderr_text = (result.get("stderr") or "").strip()
    if stderr_text:
        lines.append("")
        lines.append("Clingo stderr:")
        lines.append(stderr_text)

    logs = result.get("logs") or []
    if logs:
        lines.append("")
        lines.append("Clingo log messages:")
        lines.extend(logs)

    destination.write_text("\n".join(lines) + "\n")
