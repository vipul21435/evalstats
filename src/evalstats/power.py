"""Power and sample-size planning for paired comparisons.

The question answered here: how many items does a benchmark need before a
difference of ``delta`` between two models on the same items is detectable?
The answer depends on how often the two models disagree, not on their
scores. Two models that solve almost the same items can be told apart on a
small benchmark; two models that disagree on half the items need a large one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy import stats

from evalstats.intervals import finite_array, standard_error


@dataclass(frozen=True)
class PowerPlan:
    """Items needed to detect ``delta`` at the given error rates."""

    delta: float
    alpha: float
    power: float
    discordance: float | None
    sd_difference: float
    design_effect: float
    n_items: int
    n_items_clustered: int
    n_observed: int | None = None
    power_at_observed: float | None = None
    detectable_at_observed: float | None = None

    def to_dict(self) -> dict:
        return {
            "delta": self.delta,
            "alpha": self.alpha,
            "power": self.power,
            "discordance": self.discordance,
            "sd_difference": self.sd_difference,
            "design_effect": self.design_effect,
            "n_items": self.n_items,
            "n_items_clustered": self.n_items_clustered,
            "n_observed": self.n_observed,
            "power_at_observed": self.power_at_observed,
            "detectable_at_observed": self.detectable_at_observed,
        }


def _z(alpha: float, power: float) -> tuple[float, float]:
    if not 0 < alpha < 1 or not 0 < power < 1:
        raise ValueError("alpha and power must be in (0, 1)")
    return float(stats.norm.ppf(1 - alpha / 2)), float(stats.norm.ppf(power))


def items_for_discordance(
    delta: float, discordance: float, *, alpha: float = 0.05, power: float = 0.8
) -> int:
    """Items needed for a paired test of binary scores.

    With a share ``psi`` of discordant items and a true difference ``delta``
    the per-item difference has variance ``psi`` under the null and
    ``psi - delta**2`` under the alternative, so

        n = (z_{1-alpha/2} sqrt(psi) + z_{power} sqrt(psi - delta^2))^2 / delta^2

    which is the usual sample size for McNemar's test. ``delta`` cannot
    exceed ``discordance`` (a model cannot win by more than the share of
    items the two disagree on).
    """
    if delta <= 0:
        raise ValueError("delta must be positive")
    if not 0 < discordance <= 1:
        raise ValueError("discordance must be in (0, 1]")
    if delta > discordance:
        raise ValueError("delta cannot exceed the discordance rate")
    z_a, z_b = _z(alpha, power)
    n = (z_a * math.sqrt(discordance) + z_b * math.sqrt(discordance - delta**2)) ** 2 / delta**2
    return int(math.ceil(n))


def items_for_sd(delta: float, sd: float, *, alpha: float = 0.05, power: float = 0.8) -> int:
    """Items needed when the per-item differences have standard deviation ``sd``.

    ``n = ((z_{1-alpha/2} + z_{power}) sd / delta)^2`` for a paired t-type test.
    """
    if delta <= 0:
        raise ValueError("delta must be positive")
    if sd <= 0:
        raise ValueError("sd must be positive")
    z_a, z_b = _z(alpha, power)
    return int(math.ceil(((z_a + z_b) * sd / delta) ** 2))


def power_for_items(n: int, delta: float, sd: float, *, alpha: float = 0.05) -> float:
    """Power of a two-sided paired test with ``n`` items (normal approximation)."""
    if n <= 0 or sd <= 0:
        raise ValueError("n and sd must be positive")
    z_a = float(stats.norm.ppf(1 - alpha / 2))
    shift = abs(delta) * math.sqrt(n) / sd
    return float(stats.norm.cdf(shift - z_a) + stats.norm.cdf(-shift - z_a))


def detectable_difference(n: int, sd: float, *, alpha: float = 0.05, power: float = 0.8) -> float:
    """Smallest difference a paired test on ``n`` items detects with the given power."""
    if n <= 0 or sd <= 0:
        raise ValueError("n and sd must be positive")
    z_a, z_b = _z(alpha, power)
    return float((z_a + z_b) * sd / math.sqrt(n))


def plan_from_scores(
    a: ArrayLike,
    b: ArrayLike,
    delta: float,
    *,
    clusters: ArrayLike | None = None,
    alpha: float = 0.05,
    power: float = 0.8,
) -> PowerPlan:
    """Plan from the observed scores of two models on the same items.

    The discordance (binary scores) or the standard deviation of the per-item
    differences (real scores) is read off the data. The design effect is the
    squared ratio of the cluster-robust to the plain standard error of the
    difference; ``n_items_clustered`` inflates the plain answer by it.

    With few clusters the estimated design effect is noisy and can fall
    below 1 by chance. It is reported as estimated, but the clustered item
    count and the power on the observed items use ``max(1, design_effect)``,
    so clustering never makes the plan more optimistic than independent
    items would.
    """
    a = finite_array(a)
    b = finite_array(b)
    if a.shape != b.shape or a.size < 2:
        raise ValueError("need two aligned score vectors with at least two items")
    d = a - b
    binary = set(np.unique(a)) <= {0.0, 1.0} and set(np.unique(b)) <= {0.0, 1.0}
    sd = float(d.std(ddof=1))
    disc = float(np.mean(d != 0)) if binary else None
    if binary and disc is not None and disc > 0:
        n_plain = items_for_discordance(delta, disc, alpha=alpha, power=power)
    else:
        n_plain = items_for_sd(delta, sd, alpha=alpha, power=power) if sd > 0 else 0
    deff = 1.0
    if clusters is not None:
        se_plain = standard_error(d)
        se_cl = standard_error(d, clusters)
        if se_plain > 0 and not math.isnan(se_cl):
            deff = float((se_cl / se_plain) ** 2)
    inflation = max(1.0, deff)
    n_cl = int(math.ceil(n_plain * inflation))
    n_obs = int(a.size)
    sd_eff = sd * math.sqrt(inflation)
    pw = power_for_items(n_obs, delta, sd_eff, alpha=alpha) if sd_eff > 0 else None
    mdd = detectable_difference(n_obs, sd_eff, alpha=alpha, power=power) if sd_eff > 0 else None
    return PowerPlan(delta, alpha, power, disc, sd, deff, n_plain, n_cl, n_obs, pw, mdd)
