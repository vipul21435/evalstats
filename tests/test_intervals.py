import numpy as np
import pytest
from scipy import stats

from evalstats.intervals import mean_interval, standard_error


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
