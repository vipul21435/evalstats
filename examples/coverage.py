"""Print the coverage of item-level and cluster-robust 90 % intervals.

The same seeded simulation as
tests/test_intervals.py::test_clustered_clt_coverage_with_cluster_effects:
20 clusters of unequal size, scores ``u_g + e_i`` with a shared cluster effect
``u_g ~ N(0, 1)``, item noise ``e_i ~ N(0, 1)`` and true mean 0.

    python examples/coverage.py
"""

from __future__ import annotations

import argparse

import numpy as np

from evalstats.intervals import mean_interval

SIZES = [60, 40, 30, 30, 25, 25, 20, 20, 20, 15] + [10] * 10


def simulate(reps: int = 400, seed: int = 11, level: float = 0.90) -> tuple[float, float]:
    """Share of replications whose item-level and clustered intervals cover 0."""
    rng = np.random.default_rng(seed)
    sizes = np.array(SIZES)
    clusters = np.repeat(np.arange(sizes.size), sizes)
    hits_plain = hits_cl = 0
    for _ in range(reps):
        x = rng.normal(0, 1, sizes.size)[clusters] + rng.normal(0, 1, clusters.size)
        plain = mean_interval(x, level=level)
        cl = mean_interval(x, level=level, clusters=clusters)
        hits_plain += plain.low <= 0 <= plain.high
        hits_cl += cl.low <= 0 <= cl.high
    return hits_plain / reps, hits_cl / reps


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--reps", type=int, default=400)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()
    plain, cl = simulate(args.reps, args.seed)
    print(f"item-level 90% interval coverage: {plain:.4f}")
    print(f"cluster-robust 90% interval coverage: {cl:.4f}")


if __name__ == "__main__":
    main()
