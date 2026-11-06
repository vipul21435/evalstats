"""Markdown and JSON rendering shared by the commands."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path


def pct(x: float) -> str:
    """A proportion as a percentage with one decimal."""
    return f"{100 * x:.1f}"


def p_value(p: float) -> str:
    """A p-value with four decimals, or ``<0.0001`` below that."""
    return "<0.0001" if p < 1e-4 else f"{p:.4f}"


def _is_number(v: object) -> bool:
    if isinstance(v, bool):
        return False
    if isinstance(v, (int, float)):
        return True
    if isinstance(v, str):
        try:
            float(v.lstrip("<"))
        except ValueError:
            return False
        return True
    return False


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    """A pipe table.

    A column is right-aligned when every cell is a number (an int, a float or
    a string such as ``"12.3"`` or ``"<0.0001"``) or a missing value (``None``
    or ``"-"``), and at least one cell is a number.
    """
    cols = len(headers)
    numeric = []
    for j in range(cols):
        cells = [r[j] for r in rows]
        present = [v for v in cells if v is not None and v != "-"]
        numeric.append(bool(present) and all(_is_number(v) for v in present))
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
