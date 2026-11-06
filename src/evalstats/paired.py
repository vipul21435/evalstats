"""Paired comparison of two models scored on the same items."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy import stats

from evalstats.intervals import Interval, Method, finite_array, mean_interval, row_chunks


@dataclass(frozen=True)
class McNemar:
    """Exact McNemar test on the discordant pairs of two binary score vectors.

    ``b`` counts items model A solved and model B did not; ``c`` the reverse.
    Under the null the discordant items split as ``Binomial(b + c, 1/2)``.
    """

    b: int
    c: int
    p_value: float

    @property
    def discordant(self) -> int:
        return self.b + self.c

    def to_dict(self) -> dict[str, float | int]:
        return {"b": self.b, "c": self.c, "discordant": self.discordant, "p_value": self.p_value}


@dataclass(frozen=True)
class PairedResult:
    """Everything the paired comparison computes."""

    n: int
    mean_a: float
    mean_b: float
    difference: Interval
    permutation_p: float
    mcnemar: McNemar | None
    cluster_permutation_p: float | None = None
    n_clusters: int | None = None

    def to_dict(self) -> dict:
        return {
            "n": self.n,
            "mean_a": self.mean_a,
            "mean_b": self.mean_b,
            "difference": self.difference.to_dict(),
            "permutation_p": self.permutation_p,
            "mcnemar": None if self.mcnemar is None else self.mcnemar.to_dict(),
            "cluster_permutation_p": self.cluster_permutation_p,
            "n_clusters": self.n_clusters,
        }


def mcnemar_exact(a: ArrayLike, b: ArrayLike) -> McNemar:
    """Two-sided exact McNemar test for binary scores ``a`` and ``b``."""
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    if a.shape != b.shape:
        raise ValueError("score vectors must have the same length")
    if not (set(np.unique(a)) <= {0.0, 1.0} and set(np.unique(b)) <= {0.0, 1.0}):
        raise ValueError("McNemar needs 0/1 scores")
    n_b = int(np.sum((a == 1) & (b == 0)))
    n_c = int(np.sum((a == 0) & (b == 1)))
    m = n_b + n_c
    if m == 0:
        return McNemar(n_b, n_c, 1.0)
    k = min(n_b, n_c)
    p = min(1.0, 2.0 * stats.binom.cdf(k, m, 0.5))
    return McNemar(n_b, n_c, float(p))


def sign_flip_test(diffs: ArrayLike, *, n_perm: int = 10000, seed: int | None = 0) -> float:
    """Two-sided sign-flip permutation p-value for paired differences.

    Under the null each item's difference is as likely to have either sign,
    so the mean difference is compared with the distribution obtained by
    flipping signs at random. Returns ``(count + 1) / (n_perm + 1)``.
    """
    d = finite_array(diffs, "scores")
    if d.size == 0:
        raise ValueError("no differences")
    observed = abs(d.mean())
    rng = np.random.default_rng(seed)
    count = 0
    for m in row_chunks(n_perm, d.size):
        signs = rng.choice([-1.0, 1.0], size=(m, d.size))
        perm = np.abs((signs * d).mean(axis=1))
        count += int(np.sum(perm >= observed - 1e-12))
    return (count + 1) / (n_perm + 1)


def cluster_sign_flip_test(
    diffs: ArrayLike,
    clusters: ArrayLike,
    *,
    n_perm: int = 10000,
    max_exact: int = 16,
    seed: int | None = 0,
) -> float:
    """Two-sided sign-flip test that flips whole clusters at once.

    The per-item differences are summed within each cluster, and the null is
    that each cluster sum is as likely to have either sign. This keeps the
    dependence inside a cluster (a repository where one model does well on
    every issue counts once, not once per issue). With ``G <= max_exact``
    clusters all ``2**G`` sign patterns are enumerated and the p-value is
    exact; above that ``n_perm`` random patterns are drawn and the p-value is
    ``(count + 1) / (n_perm + 1)``.

    With ``G`` clusters the smallest attainable p-value is ``2 / 2**G``, so a
    handful of clusters cannot give a small p-value whatever the data.
    """
    d = finite_array(diffs, "scores")
    c = np.asarray(clusters).ravel()
    if d.size == 0 or c.size != d.size:
        raise ValueError("differences and clusters must align and be non-empty")
    _, inv = np.unique(c, return_inverse=True)
    g = int(inv.max()) + 1
    sums = np.bincount(inv, weights=d, minlength=g)
    observed = abs(sums.sum())
    tol = 1e-9 * max(1.0, float(np.abs(sums).sum()))
    if g <= max_exact:
        codes = np.arange(2**g)[:, None]
        signs = 1.0 - 2.0 * ((codes >> np.arange(g)) & 1)
        stat = np.abs(signs @ sums)
        return float(np.mean(stat >= observed - tol))
    rng = np.random.default_rng(seed)
    count = 0
    for m in row_chunks(n_perm, g):
        signs = rng.choice([-1.0, 1.0], size=(m, g))
        count += int(np.sum(np.abs(signs @ sums) >= observed - tol))
    return (count + 1) / (n_perm + 1)


def is_binary(x: ArrayLike) -> bool:
    return bool(set(np.unique(np.asarray(x, dtype=float))) <= {0.0, 1.0})


def paired_comparison(
    a: ArrayLike,
    b: ArrayLike,
    *,
    level: float = 0.95,
    method: Method = "clt",
    clusters: ArrayLike | None = None,
    n_boot: int = 2000,
    n_perm: int = 10000,
    seed: int | None = 0,
) -> PairedResult:
    """Compare model A with model B on the same items.

    ``a`` and ``b`` are aligned score vectors. The difference interval is a
    one-sample interval on ``a - b`` (cluster-robust when ``clusters`` is
    given). The McNemar test is run only when both vectors are binary; the
    cluster sign-flip test only when ``clusters`` is given.
    """
    a = finite_array(a)
    b = finite_array(b)
    if a.shape != b.shape:
        raise ValueError("score vectors must have the same length")
    d = a - b
    diff = mean_interval(d, level=level, method=method, clusters=clusters, n_boot=n_boot, seed=seed)
    perm_p = sign_flip_test(d, n_perm=n_perm, seed=seed)
    mc = mcnemar_exact(a, b) if is_binary(a) and is_binary(b) else None
    cl_p = None
    n_cl = None
    if clusters is not None:
        cl_p = cluster_sign_flip_test(d, clusters, n_perm=n_perm, seed=seed)
        n_cl = int(np.unique(np.asarray(clusters)).size)
    return PairedResult(int(a.size), float(a.mean()), float(b.mean()), diff, perm_p, mc, cl_p, n_cl)
