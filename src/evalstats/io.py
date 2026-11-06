"""Reading per-item result files.

A results file is JSONL or CSV with one row per (item, model) pair. Required
columns are ``item``, ``model`` and ``score``; ``cluster`` is optional. For
pass@k data a row may carry ``n_samples`` and ``n_correct`` instead of a
single score.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Record:
    item: str
    model: str
    score: float | None
    cluster: str | None = None
    n_samples: int | None = None
    n_correct: int | None = None


def _coerce(row: dict) -> Record:
    score = row.get("score")
    score_f = None if score in (None, "") else float(score)
    ns = row.get("n_samples")
    nc = row.get("n_correct")
    return Record(
        item=str(row["item"]),
        model=str(row["model"]),
        score=score_f,
        cluster=None if row.get("cluster") in (None, "") else str(row["cluster"]),
        n_samples=None if ns in (None, "") else int(ns),
        n_correct=None if nc in (None, "") else int(nc),
    )


def read_records(path: str | Path) -> list[Record]:
    """Read a JSONL (``.jsonl``/``.json``) or CSV results file."""
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if p.suffix.lower() == ".csv":
        rows: Iterable[dict] = csv.DictReader(text.splitlines())
    else:
        rows = (json.loads(line) for line in text.splitlines() if line.strip())
    return [_coerce(r) for r in rows]


@dataclass(frozen=True)
class ModelScores:
    """Scores of one model, ordered by item id."""

    model: str
    items: list[str]
    scores: np.ndarray
    clusters: np.ndarray | None


def scores_by_model(
    records: list[Record], *, default_cluster: bool = False
) -> dict[str, ModelScores]:
    """Group records by model; items are sorted so models align by item id.

    With ``default_cluster`` and no ``cluster`` column, the cluster is the
    part of the item id before the first ``__``, which is how many benchmarks
    prefix item ids with their source repository.
    """
    by_model: dict[str, dict[str, Record]] = {}
    for r in records:
        if r.score is None:
            continue
        by_model.setdefault(r.model, {})[r.item] = r
    out: dict[str, ModelScores] = {}
    for model, rows in by_model.items():
        items = sorted(rows)
        scores = np.array([rows[i].score for i in items], dtype=float)
        has_cluster = any(rows[i].cluster is not None for i in items)
        clusters: np.ndarray | None
        if has_cluster:
            clusters = np.array([rows[i].cluster or "" for i in items])
        elif default_cluster:
            clusters = np.array([i.split("__", 1)[0] for i in items])
        else:
            clusters = None
        out[model] = ModelScores(model, items, scores, clusters)
    return out


Aligned = tuple[list[str], np.ndarray, np.ndarray, "np.ndarray | None"]


def align(a: ModelScores, b: ModelScores) -> Aligned:
    """Restrict two models to their shared items, in a common order."""
    shared = sorted(set(a.items) & set(b.items))
    if not shared:
        raise ValueError(f"models {a.model!r} and {b.model!r} share no items")
    ia = {it: k for k, it in enumerate(a.items)}
    ib = {it: k for k, it in enumerate(b.items)}
    sa = np.array([a.scores[ia[i]] for i in shared])
    sb = np.array([b.scores[ib[i]] for i in shared])
    clusters = None if a.clusters is None else np.array([a.clusters[ia[i]] for i in shared])
    return shared, sa, sb, clusters
