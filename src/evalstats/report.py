"""Markdown and JSON rendering shared by the commands, and the full report."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from evalstats.io import ModelScores, align
from evalstats.leaderboard import Leaderboard, leaderboard
from evalstats.power import plan_from_scores


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


def leaderboard_tables(lb: Leaderboard) -> str:
    """The ranking table, the pairwise table and the top-group line."""
    lvl = int(round(100 * (1 - lb.alpha)))
    rank_rows = [
        [
            i + 1,
            name,
            iv.n,
            "-" if iv.n_clusters is None else iv.n_clusters,
            pct(iv.estimate),
            pct(iv.low),
            pct(iv.high),
        ]
        for i, (name, iv) in enumerate(lb.models)
    ]
    ranks = markdown_table(
        ["rank", "model", "n", "clusters", "mean %", f"{lvl}% low", f"{lvl}% high"], rank_rows
    )
    pair_rows = [
        [
            r.a,
            r.b,
            r.n,
            pct(r.difference),
            pct(r.low),
            pct(r.high),
            r.test,
            p_value(r.p_value),
            p_value(r.p_holm),
            p_value(r.p_bh),
        ]
        for r in lb.pairs
    ]
    pairs = markdown_table(
        ["a", "b", "n", "a - b", f"{lvl}% low", f"{lvl}% high", "test", "p", "p Holm", "p BH"],
        pair_rows,
    )
    top = ", ".join(lb.top_group())
    return (
        f"{ranks}\n\n{pairs}\n\n"
        f"not separated from the leader at Holm-adjusted alpha {lb.alpha:g}: {top}"
    )


def neighbour_plans(
    models: dict[str, ModelScores], lb: Leaderboard, delta: float, *, power: float = 0.8
) -> list[dict]:
    """A power plan for each pair of models adjacent in the ranking."""
    names = [n for n, _ in lb.models]
    out = []
    for a, b in zip(names, names[1:], strict=False):
        _, sa, sb, clusters = align(models[a], models[b])
        try:
            plan = plan_from_scores(sa, sb, delta, clusters=clusters, alpha=lb.alpha, power=power)
        except ValueError:
            # delta larger than the observed discordance, or two models that
            # never disagree: the data give no finite plan
            continue
        out.append({"a": a, "b": b, **plan.to_dict()})
    return out


def full_report(
    models: dict[str, ModelScores],
    *,
    source: str,
    delta: float = 0.03,
    power: float = 0.8,
    **leaderboard_args,
) -> dict:
    """Everything in one payload: intervals, corrected pairs, power plans."""
    lb = leaderboard(models, **leaderboard_args)
    items = sorted({i for m in models.values() for i in m.items})
    return {
        "source": source,
        "n_models": len(models),
        "n_items": len(items),
        "leaderboard": lb,
        "plans": neighbour_plans(models, lb, delta, power=power),
        "delta": delta,
        "power": power,
    }


def render_markdown(rep: dict) -> str:
    """The report as one Markdown document."""
    lb: Leaderboard = rep["leaderboard"]
    lines = [
        "# evalstats report",
        "",
        f"Source: `{rep['source']}`; {rep['n_models']} models, {rep['n_items']} distinct items.",
        "",
        "## Intervals and pairwise tests",
        "",
        leaderboard_tables(lb),
        "",
        f"## Items needed to detect {pct(rep['delta'])} points between neighbours",
        "",
    ]
    rows = [
        [
            p["a"],
            p["b"],
            "-" if p["discordance"] is None else pct(p["discordance"]),
            f"{p['design_effect']:.2f}",
            p["n_items"],
            p["n_items_clustered"],
            "-" if p["power_at_observed"] is None else f"{p['power_at_observed']:.2f}",
        ]
        for p in rep["plans"]
    ]
    lines.append(
        markdown_table(
            [
                "a",
                "b",
                "discordant %",
                "design effect",
                "items",
                "items (clustered)",
                f"power on {rep['n_items']}",
            ],
            rows,
        )
    )
    lines += [
        "",
        f"Power target {rep['power']:g}, two-sided alpha {lb.alpha:g}. A pair is left out when "
        "the difference exceeds the share of items the two models disagree on, or when the "
        "two models agree on every item (or their differences do not vary), so that the "
        "data give no plan.",
    ]
    return "\n".join(lines)


def report_json(rep: dict) -> str:
    payload = dict(rep)
    payload["leaderboard"] = rep["leaderboard"].to_dict()
    return to_json(payload)
