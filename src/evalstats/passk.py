"""Unbiased pass@k over items with several samples each."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from evalstats.intervals import Interval, row_chunks


def pass_at_k_item(n: int, c: int, k: int) -> float:
    """Unbiased pass@k for one item with ``c`` correct out of ``n`` samples.

    ``1 - C(n - c, k) / C(n, k)``, computed as a product to avoid large
    binomials. Returns 1.0 when fewer than ``k`` samples are wrong.
    Raises ``ValueError`` unless ``1 <= k <= n`` and ``0 <= c <= n``.
    """
    if k < 1:
        raise ValueError("k must be at least 1")
    if not 0 <= c <= n:
        raise ValueError(f"n_correct must be between 0 and n_samples, got {c} of {n}")
    if k > n:
        raise ValueError("k must not exceed the number of samples")
    if n - c < k:
        return 1.0
    return float(1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1)))


def pass_at_k(
    n_samples: ArrayLike,
    n_correct: ArrayLike,
    k: int,
    *,
    level: float = 0.95,
    n_boot: int = 2000,
    seed: int | None = 0,
) -> Interval:
    """Mean pass@k over items with a percentile-bootstrap interval over items.

    The per-item estimator is unbiased; the interval resamples items, not
    samples, so it reflects how much the benchmark's items disagree.
    """
    n = np.asarray(n_samples, dtype=int).ravel()
    c = np.asarray(n_correct, dtype=int).ravel()
    if n.shape != c.shape or n.size == 0:
        raise ValueError("n_samples and n_correct must align and be non-empty")
    per_item = np.array([pass_at_k_item(int(ni), int(ci), k) for ni, ci in zip(n, c, strict=True)])
    est = float(per_item.mean())
    rng = np.random.default_rng(seed)
    size = per_item.size
    boot = np.concatenate(
        [
            per_item[rng.integers(0, size, size=(m, size))].mean(axis=1)
            for m in row_chunks(n_boot, size)
        ]
    )
    lo, hi = np.percentile(boot, [100 * (0.5 - level / 2), 100 * (0.5 + level / 2)])
    return Interval(est, float(lo), float(hi), level, "percentile", int(per_item.size))
