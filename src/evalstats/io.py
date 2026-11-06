"""Reading per-item result files.

A results file is JSONL or CSV with one row per (item, model) pair. Required
columns are ``item``, ``model`` and ``score``; ``cluster`` is optional. For
pass@k data a row may carry ``n_samples`` and ``n_correct`` instead of a
single score.
"""

from __future__ import annotations

import csv
import json
import math
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


def _number(row: dict, key: str, kind: type, where: str):
    v = row.get(key)
    if v in (None, ""):
        return None
    try:
        out = kind(v)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{where}: column {key!r} is not a number: {v!r}") from None
    if not math.isfinite(out):
        raise ValueError(f"{where}: column {key!r} is not a finite number: {v!r}")
    return out


def _coerce(row: object, where: str) -> Record:
    if not isinstance(row, dict):
        raise ValueError(f"{where}: expected an object with item, model and score")
    for key in ("item", "model"):
        if row.get(key) in (None, ""):
            raise ValueError(f"{where}: missing column {key!r}")
    return Record(
        item=str(row["item"]),
        model=str(row["model"]),
        score=_number(row, "score", float, where),
        cluster=None if row.get("cluster") in (None, "") else str(row["cluster"]),
        n_samples=_number(row, "n_samples", int, where),
        n_correct=_number(row, "n_correct", int, where),
    )


def read_records(path: str | Path) -> list[Record]:
    """Read a JSONL (``.jsonl``/``.json``) or CSV results file, UTF-8 encoded.

    A malformed row raises ``ValueError`` naming the file and line number.
    """
    p = Path(path)
    try:
        # utf-8-sig drops the byte-order mark some spreadsheet tools write
        text = p.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        raise ValueError(f"{p}: not UTF-8 text") from None
    out: list[Record] = []
    if p.suffix.lower() == ".csv":
        reader = csv.DictReader(text.splitlines())
        for row in reader:
            out.append(_coerce(row, f"{p}:{reader.line_num}"))
        return out
    for lineno, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{p}:{lineno}: not valid JSON ({exc.msg})") from None
        out.append(_coerce(row, f"{p}:{lineno}"))
    return out


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

    When a model has several rows for the same item (several samples), the
    item's score is the mean of those rows. The item, not the sample, stays
    the unit of resampling, so repeated samples narrow an interval only as
    far as they reduce the noise in each item's score.

    With ``default_cluster`` and no ``cluster`` column, the cluster is the
    part of the item id before the first ``__``, which is how many benchmarks
    prefix item ids with their source repository.
    """
    by_model: dict[str, dict[str, list[Record]]] = {}
    for r in records:
        if r.score is None:
            continue
        by_model.setdefault(r.model, {}).setdefault(r.item, []).append(r)
    out: dict[str, ModelScores] = {}
    for model, rows in by_model.items():
        items = sorted(rows)
        # several rows for one (model, item) are samples: score the item by
        # their mean, which for 0/1 scores is the per-item pass@1 estimate
        scores = np.array(
            [float(np.mean([r.score for r in rows[i]])) for i in items], dtype=float
        )
        first = {i: rows[i][0] for i in items}
        has_cluster = any(first[i].cluster is not None for i in items)
        clusters: np.ndarray | None
        if has_cluster:
            clusters = np.array([first[i].cluster or "" for i in items])
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


def passk_counts(records: list[Record]) -> dict[str, list[tuple[int, int]]]:
    """Per model, ``(n_samples, n_correct)`` for each item.

    A row carrying ``n_samples`` and ``n_correct`` is taken as is. Otherwise
    every scored row of the same (model, item) counts as one sample, so a
    file with one row per sample works too; scores must then be 0/1.
    """
    direct: dict[str, dict[str, tuple[int, int]]] = {}
    samples: dict[str, dict[str, list[float]]] = {}
    for r in records:
        if r.n_samples is not None and r.n_correct is not None:
            direct.setdefault(r.model, {})[r.item] = (r.n_samples, r.n_correct)
        elif r.score is not None:
            if r.score not in (0.0, 1.0):
                raise ValueError("per-sample pass@k rows need 0/1 scores")
            samples.setdefault(r.model, {}).setdefault(r.item, []).append(r.score)
    out: dict[str, list[tuple[int, int]]] = {}
    for model, items in direct.items():
        out[model] = [items[i] for i in sorted(items)]
    for model, items in samples.items():
        if model in out:
            continue
        out[model] = [(len(items[i]), int(sum(items[i]))) for i in sorted(items)]
    return out
