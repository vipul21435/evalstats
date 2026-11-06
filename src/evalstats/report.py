"""Markdown and JSON rendering shared by the commands."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path


def pct(x: float) -> str:
    """A proportion as a percentage with one decimal."""
    return f"{100 * x:.1f}"


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    """A pipe table; numeric columns (every cell an int or float) are right-aligned."""
    cols = len(headers)
    numeric = [
        all(isinstance(r[j], (int, float)) and not isinstance(r[j], bool) for r in rows)
        and bool(rows)
        for j in range(cols)
    ]
    align = ["---:" if numeric[j] else "---" for j in range(cols)]
    lines = ["| " + " | ".join(str(h) for h in headers) + " |", "|" + "|".join(align) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(_cell(v) for v in r) + " |")
    return "\n".join(lines)


def _cell(v: object) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.4f}" if abs(v) < 1 else f"{v:.1f}"
    return str(v)


def emit(text: str, out: str | Path | None = None) -> None:
    """Print ``text`` or write it to ``out`` when given."""
    if out is None:
        print(text)
    else:
        Path(out).write_text(text + "\n", encoding="utf-8")


def to_json(payload: object) -> str:
    return json.dumps(payload, indent=2)
