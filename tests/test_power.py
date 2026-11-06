import math

import numpy as np
import pytest
from scipy import stats

from evalstats.paired import mcnemar_exact
from evalstats.power import (
    detectable_difference,
    items_for_discordance,
    items_for_sd,
    plan_from_scores,
    power_for_items,
)


def test_items_for_discordance_closed_form():
    # psi = 0.2, delta = 0.05, alpha 0.05, power 0.8: hand computation of the formula
    z_a, z_b = stats.norm.ppf(0.975), stats.norm.ppf(0.8)
    n = (z_a * math.sqrt(0.2) + z_b * math.sqrt(0.2 - 0.0025)) ** 2 / 0.0025
    assert items_for_discordance(0.05, 0.2) == math.ceil(n)
    assert items_for_discordance(0.05, 0.2) == 626
    # fewer disagreements -> fewer items for the same gap
    assert items_for_discordance(0.03, 0.1) < items_for_discordance(0.03, 0.4)


def test_items_for_sd_matches_detectable_difference():
    n = items_for_sd(0.03, 0.5)
    # the smallest detectable difference on n items is at most delta, and on
    # n - 1 items it is larger than delta
    assert detectable_difference(n, 0.5) <= 0.03 + 1e-12
    assert detectable_difference(n - 1, 0.5) > 0.03


def test_power_increases_with_n_and_hits_target():
    sd = 0.6
    n = items_for_sd(0.05, sd, power=0.8)
    assert 0.8 <= power_for_items(n, 0.05, sd) < 0.82
    assert power_for_items(10, 0.05, sd) < power_for_items(1000, 0.05, sd)
    # size: with delta 0 the "power" is alpha
    assert abs(power_for_items(100, 0.0, sd, alpha=0.05) - 0.05) < 1e-9


def test_invalid_inputs():
    with pytest.raises(ValueError):
        items_for_discordance(0.0, 0.2)
    with pytest.raises(ValueError):
        items_for_discordance(0.3, 0.2)
    with pytest.raises(ValueError):
        items_for_discordance(0.1, 0.0)
    with pytest.raises(ValueError):
        items_for_sd(0.1, 0.0)
    with pytest.raises(ValueError):
        items_for_sd(0.1, 0.5, alpha=1.5)
    with pytest.raises(ValueError):
        power_for_items(0, 0.1, 0.5)
    with pytest.raises(ValueError):
        detectable_difference(5, 0.0)
    with pytest.raises(ValueError):
        plan_from_scores([1], [0], 0.1)


def test_plan_from_binary_scores_uses_discordance():
    rng = np.random.default_rng(3)
    a = (rng.random(200) < 0.5).astype(float)
    flip = rng.random(200) < 0.25
    b = np.where(flip, 1 - a, a)
    plan = plan_from_scores(a, b, 0.03)
    assert plan.discordance == pytest.approx(flip.mean())
    assert plan.n_items == items_for_discordance(0.03, float(flip.mean()))
    assert plan.design_effect == 1.0 and plan.n_items_clustered == plan.n_items
    assert 0 < plan.power_at_observed < 1
    assert plan.detectable_at_observed > 0.03
    assert set(plan.to_dict()) >= {"n_items", "n_items_clustered", "discordance"}


def test_plan_with_clusters_inflates_by_design_effect():
    rng = np.random.default_rng(5)
    clusters = np.repeat(np.arange(10), 20)
    shift = rng.normal(0, 0.3, size=10)[clusters]
    a = rng.normal(0.5, 0.2, size=200) + shift
    b = rng.normal(0.5, 0.2, size=200)
    plan = plan_from_scores(a, b, 0.05, clusters=clusters)
    assert plan.discordance is None
    assert plan.design_effect > 1
    assert plan.n_items_clustered == math.ceil(plan.n_items * plan.design_effect)


def test_design_effect_below_one_does_not_shrink_the_plan():
    # differences alternate in sign inside each cluster, so cluster sums are
    # near zero and the estimated design effect is far below 1
    clusters = np.repeat(np.arange(10), 20)
    a = np.tile([1.0, 0.0], 100)
    b = np.zeros(200)
    b[::4] = 1.0
    plan = plan_from_scores(a, b, 0.05, clusters=clusters)
    assert plan.design_effect < 1
    assert plan.n_items_clustered == plan.n_items
    plain = plan_from_scores(a, b, 0.05)
    assert plan.power_at_observed == pytest.approx(plain.power_at_observed)


def test_plan_is_undefined_when_the_models_never_disagree():
    # zero discordance would give "0 items", which reads as "no data needed"
    with pytest.raises(ValueError, match="agree on every shared item"):
        plan_from_scores([1, 0, 1, 0], [1, 0, 1, 0], 0.1)
    # real scores whose differences do not vary give no spread to plan from
    with pytest.raises(ValueError, match="do not vary"):
        plan_from_scores([0.5, 0.2, 0.9], [0.4, 0.1, 0.8], 0.1)


def test_planned_size_reaches_target_power_in_simulation():
    """Seeded simulation: benchmarks of the planned size, scored by two models
    that disagree on 12 % of items and differ by 3 points, reject the null
    with the exact McNemar test close to 80 % of the time. The exact test is
    slightly conservative, so the empirical power sits a little under target."""
    psi, delta = 0.12, 0.03
    n = items_for_discordance(delta, psi)
    rng = np.random.default_rng(0)
    probs = [(psi + delta) / 2, (psi - delta) / 2, 1 - psi]
    reps, rejections = 600, 0
    for _ in range(reps):
        only_a, only_b, _ = rng.multinomial(n, probs)
        a = np.r_[np.ones(only_a), np.zeros(only_b)]
        b = np.r_[np.zeros(only_a), np.ones(only_b)]
        rejections += mcnemar_exact(a, b).p_value < 0.05
    assert 0.72 < rejections / reps < 0.86


def test_design_effect_and_clustered_power_by_hand():
    # binary scores with a shared per-cluster effect: an intraclass
    # correlation well above zero, so the design effect is clearly above 1
    rng = np.random.default_rng(12)
    g, m = 25, 16
    clusters = np.repeat(np.arange(g), m)
    p_a = np.clip(0.5 + rng.normal(0, 0.25, g), 0.05, 0.95)[clusters]
    a = (rng.random(g * m) < p_a).astype(float)
    b = (rng.random(g * m) < 0.45).astype(float)
    d = a - b
    n = d.size
    se_plain = d.std(ddof=1) / math.sqrt(n)
    resid_sums = np.array([np.sum(d[clusters == k] - d.mean()) for k in range(g)])
    se_cl = math.sqrt(g / (g - 1) * np.sum(resid_sums**2) / n**2)
    deff = (se_cl / se_plain) ** 2
    assert deff > 1.3
    plan = plan_from_scores(a, b, 0.05, clusters=clusters)
    assert plan.design_effect == pytest.approx(deff, rel=1e-12)
    assert plan.n_items_clustered == math.ceil(plan.n_items * deff)
    sd = d.std(ddof=1)
    assert plan.power_at_observed == pytest.approx(
        power_for_items(n, 0.05, sd * math.sqrt(deff)), rel=1e-12
    )
    assert plan.detectable_at_observed == pytest.approx(
        detectable_difference(n, sd * math.sqrt(deff)), rel=1e-12
    )
    plain = plan_from_scores(a, b, 0.05)
    assert plan.power_at_observed < plain.power_at_observed
