import numpy as np
import pytest
from scipy import stats

from evalstats.intervals import critical_value, mean_interval, standard_error


def test_clt_matches_closed_form():
    x = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 1], dtype=float)
    iv = mean_interval(x, level=0.95)
    se = x.std(ddof=1) / np.sqrt(10)
    z = stats.norm.ppf(0.975)
    assert iv.estimate == pytest.approx(0.6)
    assert iv.stderr == pytest.approx(se)
    assert iv.low == pytest.approx(0.6 - z * se)
    assert iv.high == pytest.approx(0.6 + z * se)
    assert iv.n == 10


def test_clustered_se_with_singleton_clusters_equals_plain_se():
    rng = np.random.default_rng(0)
    x = rng.random(40)
    plain = standard_error(x)
    clustered = standard_error(x, clusters=np.arange(40))
    # sandwich uses G/(G-1) with G = n, plain uses 1/(n-1): identical here
    assert clustered == pytest.approx(plain)


def test_clustered_se_grows_with_within_cluster_correlation():
    rng = np.random.default_rng(1)
    g = 30
    cluster_effect = rng.normal(0, 1, size=g)
    clusters = np.repeat(np.arange(g), 10)
    x = cluster_effect[clusters] + rng.normal(0, 0.3, size=g * 10)
    assert standard_error(x, clusters) > 2 * standard_error(x)


def test_bootstrap_methods_bracket_estimate():
    rng = np.random.default_rng(2)
    x = rng.random(100) < 0.3
    for method in ("percentile", "bca"):
        iv = mean_interval(x, method=method, n_boot=500)
        assert iv.low <= iv.estimate <= iv.high
        assert iv.method == method


def test_bca_degenerate_sample_collapses():
    iv = mean_interval(np.ones(20), method="bca", n_boot=200)
    assert iv.low == iv.high == 1.0


def test_unknown_method_and_empty_input():
    with pytest.raises(ValueError):
        mean_interval([0, 1], method="nope")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        mean_interval([])
    with pytest.raises(ValueError):
        standard_error([0, 1, 1], clusters=[0, 1])


@pytest.mark.parametrize("method", ["clt", "percentile", "bca"])
def test_coverage_of_binary_mean(method):
    """Seeded simulation: nominal 90% intervals should cover roughly 90%."""
    rng = np.random.default_rng(123)
    p, n, reps = 0.35, 80, 300
    hits = 0
    for r in range(reps):
        x = rng.random(n) < p
        iv = mean_interval(x, level=0.90, method=method, n_boot=300, seed=r)
        hits += iv.low <= p <= iv.high
    assert abs(hits / reps - 0.90) < 0.06


def test_cluster_bootstrap_reports_cluster_count():
    rng = np.random.default_rng(3)
    x = rng.random(60)
    clusters = np.repeat(["r1", "r2", "r3", "r4", "r5", "r6"], 10)
    iv = mean_interval(x, method="percentile", clusters=clusters, n_boot=200)
    assert iv.n_clusters == 6
    assert iv.low < iv.estimate < iv.high


def test_clustered_clt_uses_t_with_g_minus_one_df():
    rng = np.random.default_rng(7)
    x = rng.random(60)
    clusters = np.repeat(np.arange(6), 10)
    iv = mean_interval(x, level=0.95, clusters=clusters)
    se = standard_error(x, clusters)
    half = stats.t.ppf(0.975, df=5) * se
    assert iv.high - iv.estimate == pytest.approx(half)
    assert critical_value(0.95) == pytest.approx(stats.norm.ppf(0.975))
    assert critical_value(0.95, 6) > critical_value(0.95, 60) > critical_value(0.95)
    with pytest.raises(ValueError):
        critical_value(1.5)


def test_clustered_clt_coverage_with_cluster_effects():
    """Seeded simulation: 20 unequal clusters sharing a cluster effect.

    Scores are ``u_g + e_i`` with true mean 0. The item-level interval
    ignores the dependence and under-covers badly; the cluster-robust t
    interval stays near the nominal 90 %.
    """
    rng = np.random.default_rng(11)
    sizes = np.array([60, 40, 30, 30, 25, 25, 20, 20, 20, 15] + [10] * 10)
    clusters = np.repeat(np.arange(sizes.size), sizes)
    reps, hits_plain, hits_cl = 400, 0, 0
    for _ in range(reps):
        x = rng.normal(0, 1, sizes.size)[clusters] + rng.normal(0, 1, clusters.size)
        plain = mean_interval(x, level=0.90)
        cl = mean_interval(x, level=0.90, clusters=clusters)
        hits_plain += plain.low <= 0 <= plain.high
        hits_cl += cl.low <= 0 <= cl.high
    assert hits_plain / reps < 0.6
    assert abs(hits_cl / reps - 0.90) < 0.06


def test_bca_matches_scipy_on_binary_sample():
    # 32 successes out of 100: many bootstrap means tie with the estimate,
    # which is where the bias correction must count ties at half weight.
    x = np.zeros(100)
    x[:32] = 1.0
    ours = mean_interval(x, method="bca", n_boot=20000, seed=1)
    ref = stats.bootstrap(
        (x,), np.mean, method="BCa", n_resamples=20000, confidence_level=0.95, random_state=2
    ).confidence_interval
    # the bootstrap distribution lives on a grid of 1/n = 0.01
    assert ours.low == pytest.approx(ref.low, abs=0.01)
    assert ours.high == pytest.approx(ref.high, abs=0.01)
    assert ours.low < 0.32 < ours.high
