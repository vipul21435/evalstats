"""Leaderboard mode: every model with an interval, every pair with a corrected p-value."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
from numpy.typing import ArrayLike

from evalstats.intervals import Interval, Method, mean_interval
from evalstats.io import ModelScores, align
from evalstats.paired import cluster_sign_flip_test, is_binary, mcnemar_exact, sign_flip_test

TESTS = ("auto", "mcnemar", "sign-flip", "cluster")


def holm(p_values: ArrayLike) -> np.ndarray:
    """Holm step-down adjusted p-values (control the family-wise error rate).

    Sort the ``m`` p-values; the ``i``-th smallest is multiplied by
    ``m - i`` (``i`` from 0), the sequence is made monotone with a running
    maximum, and everything is capped at 1.
    """
    p = np.asarray(p_values, dtype=float).ravel()
    m = p.size
    if m == 0:
        return p
    order = np.argsort(p)
    scaled = p[order] * (m - np.arange(m))
    adjusted = np.minimum(1.0, np.maximum.accumulate(scaled))
    out = np.empty(m)
    out[order] = adjusted
    return out


def benjamini_hochberg(p_values: ArrayLike) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (control the false discovery rate).

    Sort the ``m`` p-values; the ``i``-th smallest (``i`` from 1) is
    multiplied by ``m / i``, the sequence is made monotone from the top with
    a running minimum, and everything is capped at 1.
    """
    p = np.asarray(p_values, dtype=float).ravel()
    m = p.size
    if m == 0:
        return p
    order = np.argsort(p)
    ranks = np.arange(1, m + 1)
    scaled = p[order] * m / ranks
    adjusted = np.minimum(1.0, np.minimum.accumulate(scaled[::-1])[::-1])
    out = np.empty(m)
    out[order] = adjusted
    return out


@dataclass(frozen=True)
class PairRow:
    a: str
    b: str
    n: int
    difference: float
    low: float
    high: float
    test: str
    p_value: float
    p_holm: float
    p_bh: float

    def to_dict(self) -> dict:
        return {
            "a": self.a,
            "b": self.b,
            "n": self.n,
            "difference": self.difference,
            "low": self.low,
            "high": self.high,
            "test": self.test,
            "p_value": self.p_value,
            "p_holm": self.p_holm,
            "p_bh": self.p_bh,
        }


@dataclass(frozen=True)
class Leaderboard:
    models: list[tuple[str, Interval]]
    pairs: list[PairRow]
    alpha: float

    def top_group(self) -> list[str]:
        """Models not separated from the leader after the Holm correction."""
        if not self.models:
            return []
        leader = self.models[0][0]
        beaten = {
            r.b if r.a == leader else r.a
            for r in self.pairs
            if leader in (r.a, r.b) and r.p_holm < self.alpha
        }
        return [name for name, _ in self.models if name not in beaten]

    def to_dict(self) -> dict:
        return {
            "alpha": self.alpha,
            "models": [{"model": n, **iv.to_dict()} for n, iv in self.models],
            "pairs": [r.to_dict() for r in self.pairs],
            "top_group": self.top_group(),
        }


def leaderboard(
    models: dict[str, ModelScores],
    *,
    level: float = 0.95,
    method: Method = "clt",
    n_boot: int = 2000,
    n_perm: int = 10000,
    seed: int | None = 0,
    against_leader: bool = False,
    test: str = "auto",
) -> Leaderboard:
    """Rank models by mean score and test every pair on their shared items.

    ``test`` chooses the per-pair test: ``auto`` runs the exact McNemar test
    when both models have 0/1 scores and the item-level sign-flip test
    otherwise; ``mcnemar`` and ``sign-flip`` force one of them; ``cluster``
    runs the cluster-level sign-flip test, which needs cluster keys. Each
    pair also gets an interval on the paired difference, cluster-robust when
    clusters are known. The raw p-values of the whole family of pairs are
    then adjusted by Holm and by Benjamini-Hochberg. With ``against_leader``
    only the pairs involving the top model are tested, a smaller family.
    """
    if test not in TESTS:
        raise ValueError(f"unknown test {test!r}; choose from {', '.join(TESTS)}")
    ranked: list[tuple[str, Interval]] = []
    for name, m in models.items():
        iv = mean_interval(
            m.scores, level=level, method=method, clusters=m.clusters, n_boot=n_boot, seed=seed
        )
        ranked.append((name, iv))
    ranked.sort(key=lambda t: (-t[1].estimate, t[0]))
    names = [n for n, _ in ranked]
    if against_leader and names:
        pairs = [(names[0], other) for other in names[1:]]
    else:
        pairs = list(combinations(names, 2))
    raw: list[tuple[str, str, int, Interval, str, float]] = []
    for a, b in pairs:
        _, sa, sb, clusters = align(models[a], models[b])
        iv = mean_interval(
            sa - sb, level=level, method=method, clusters=clusters, n_boot=n_boot, seed=seed
        )
        binary = is_binary(sa) and is_binary(sb)
        name = test
        if test == "auto":
            name = "mcnemar" if binary else "sign-flip"
        if name == "mcnemar":
            p = mcnemar_exact(sa, sb).p_value
        elif name == "sign-flip":
            p = sign_flip_test(sa - sb, n_perm=n_perm, seed=seed)
        else:
            if clusters is None:
                raise ValueError(
                    "the cluster test needs cluster keys: add a cluster column "
                    "or pass --cluster-by-prefix"
                )
            p = cluster_sign_flip_test(sa - sb, clusters, n_perm=n_perm, seed=seed)
        raw.append((a, b, int(sa.size), iv, name, p))
    ps = np.array([r[5] for r in raw])
    ph = holm(ps)
    pb = benjamini_hochberg(ps)
    rows = [
        PairRow(a, b, n, iv.estimate, iv.low, iv.high, t, float(p), float(h), float(q))
        for (a, b, n, iv, t, p), h, q in zip(raw, ph, pb, strict=True)
    ]
    return Leaderboard(ranked, rows, 1 - level)
