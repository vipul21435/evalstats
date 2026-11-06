"""Confidence intervals for a mean score over evaluation items."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike
from scipy import stats

Method = Literal["clt", "percentile", "bca"]


@dataclass(frozen=True)
class Interval:
    """A point estimate with a two-sided confidence interval."""

    estimate: float
    low: float
    high: float
    level: float
    method: str
    n: int
    stderr: float | None = None
    n_clusters: int | None = None

    def to_dict(self) -> dict[str, float | int | str | None]:
        return {
            "estimate": self.estimate,
            "low": self.low,
            "high": self.high,
            "level": self.level,
            "method": self.method,
            "n": self.n,
            "stderr": self.stderr,
            "n_clusters": self.n_clusters,
        }


def finite_array(values: ArrayLike, what: str = "scores") -> np.ndarray:
    """``values`` as a flat float array; raise ``ValueError`` on NaN or inf.

    A NaN compares false with everything, so a single one would silently
    turn a permutation count into zero and a p-value into its minimum.
    """
    x = np.asarray(values, dtype=float).ravel()
    if not np.isfinite(x).all():
        raise ValueError(f"{what} must be finite")
    return x


def _as_array(scores: ArrayLike) -> np.ndarray:
    x = finite_array(scores)
    if x.size == 0:
        raise ValueError("no scores")
    return x


def standard_error(scores: ArrayLike, clusters: ArrayLike | None = None) -> float:
    """Standard error of the mean, cluster-robust when ``clusters`` is given.

    With clusters the estimator is the usual sandwich form: the variance of
    cluster-level sums of residuals, with the ``G / (G - 1)`` small-sample
    factor. With a single item per cluster it reduces to ``sd / sqrt(n)``
    up to the ``n / (n - 1)`` factor, so the two paths agree exactly.
    """
    x = _as_array(scores)
    n = x.size
    if n < 2:
        return float("nan")
    if clusters is None:
        return float(x.std(ddof=1) / np.sqrt(n))
    c = np.asarray(clusters).ravel()
    if c.size != n:
        raise ValueError("clusters must align with scores")
    resid = x - x.mean()
    _, inv = np.unique(c, return_inverse=True)
    g = int(inv.max()) + 1
    if g < 2:
        return float("nan")
    sums = np.bincount(inv, weights=resid, minlength=g)
    var = (g / (g - 1)) * np.sum(sums**2) / n**2
    return float(np.sqrt(var))


def critical_value(level: float, n_clusters: int | None = None) -> float:
    """Two-sided critical value for a CLT interval.

    Without clusters this is the normal quantile. With ``G`` clusters the
    cluster-robust variance is an average of only ``G`` cluster sums, so the
    studentised mean is closer to a t distribution with ``G - 1`` degrees of
    freedom; using the normal quantile with a dozen clusters gives intervals
    that are too short.
    """
    if not 0 < level < 1:
        raise ValueError("level must be in (0, 1)")
    q = 0.5 + level / 2
    if n_clusters is None or n_clusters < 2:
        return float(stats.norm.ppf(q))
    return float(stats.t.ppf(q, df=n_clusters - 1))


def mean_interval(
    scores: ArrayLike,
    *,
    level: float = 0.95,
    method: Method = "clt",
    clusters: ArrayLike | None = None,
    n_boot: int = 2000,
    seed: int | None = 0,
) -> Interval:
    """Interval for the mean of ``scores``.

    ``clt`` uses a normal interval on the plain standard error, or a
    t(G - 1) interval on the cluster-robust one when ``clusters`` is given.
    ``percentile`` and ``bca`` are nonparametric bootstraps over items; with
    clusters they resample whole clusters so that the dependence inside a
    cluster is preserved.
    """
    x = _as_array(scores)
    n = x.size
    est = float(x.mean())
    c = None if clusters is None else np.asarray(clusters).ravel()
    n_clusters = None if c is None else int(np.unique(c).size)

    if method == "clt":
        se = standard_error(x, c)
        z = critical_value(level, n_clusters)
        return Interval(
            est, float(est - z * se), float(est + z * se), level, "clt", n, se, n_clusters
        )

    boot = _bootstrap_means(x, c, n_boot, seed)
    if method == "percentile":
        lo, hi = np.percentile(boot, [100 * (0.5 - level / 2), 100 * (0.5 + level / 2)])
        return Interval(est, float(lo), float(hi), level, "percentile", n, None, n_clusters)
    if method == "bca":
        lo, hi = _bca_bounds(x, c, boot, est, level)
        return Interval(est, lo, hi, level, "bca", n, None, n_clusters)
    raise ValueError(f"unknown method {method!r}")


def _bootstrap_means(x: np.ndarray, c: np.ndarray | None, n_boot: int, seed: int | None):
    rng = np.random.default_rng(seed)
    n = x.size
    if c is None:
        idx = rng.integers(0, n, size=(n_boot, n))
        return x[idx].mean(axis=1)
    _, inv = np.unique(c, return_inverse=True)
    g = int(inv.max()) + 1
    sums = np.bincount(inv, weights=x, minlength=g)
    sizes = np.bincount(inv, minlength=g).astype(float)
    pick = rng.integers(0, g, size=(n_boot, g))
    return sums[pick].sum(axis=1) / sizes[pick].sum(axis=1)


def _jackknife_means(x: np.ndarray, c: np.ndarray | None) -> np.ndarray:
    n = x.size
    total = x.sum()
    if c is None:
        return (total - x) / (n - 1)
    _, inv = np.unique(c, return_inverse=True)
    g = int(inv.max()) + 1
    sums = np.bincount(inv, weights=x, minlength=g)
    sizes = np.bincount(inv, minlength=g).astype(float)
    return (total - sums) / (n - sizes)


def _bca_bounds(x, c, boot, est, level) -> tuple[float, float]:
    """Bias-corrected and accelerated bounds (Efron 1987).

    The bias correction ``z0`` is the normal quantile of the share of
    bootstrap means below the estimate, with ties counted at half weight.
    With 0/1 scores many resampled means equal the estimate exactly; counting
    them as "not below" would push ``z0`` and the whole interval down.
    """
    alpha = 1 - level
    prop = (np.mean(boot < est) + np.mean(boot <= est)) / 2
    if prop <= 0 or prop >= 1:
        # Degenerate bootstrap (for example all scores equal): fall back to
        # the percentile bounds, which collapse to the estimate.
        lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        return float(lo), float(hi)
    z0 = stats.norm.ppf(prop)
    jack = _jackknife_means(x, c)
    d = jack.mean() - jack
    denom = 6.0 * np.sum(d**2) ** 1.5
    a = 0.0 if denom == 0 else float(np.sum(d**3) / denom)
    z_lo = stats.norm.ppf(alpha / 2)
    z_hi = stats.norm.ppf(1 - alpha / 2)
    p_lo = stats.norm.cdf(z0 + (z0 + z_lo) / (1 - a * (z0 + z_lo)))
    p_hi = stats.norm.cdf(z0 + (z0 + z_hi) / (1 - a * (z0 + z_hi)))
    lo, hi = np.percentile(boot, [100 * p_lo, 100 * p_hi])
    return float(lo), float(hi)
