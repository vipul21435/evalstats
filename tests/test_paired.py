import numpy as np
import pytest

from evalstats.paired import (
    cluster_sign_flip_test,
    mcnemar_exact,
    paired_comparison,
    sign_flip_test,
)
from evalstats.passk import pass_at_k, pass_at_k_item


def test_mcnemar_hand_worked_table():
    # 20 items: both solve 6, A only 7, B only 2, neither 5.
    a = [1] * 6 + [1] * 7 + [0] * 2 + [0] * 5
    b = [1] * 6 + [0] * 7 + [1] * 2 + [0] * 5
    res = mcnemar_exact(a, b)
    assert (res.b, res.c, res.discordant) == (7, 2, 9)
    # P(X <= 2 | n=9, 1/2) = (1 + 9 + 36) / 512 = 46/512; two-sided doubles it.
    assert res.p_value == pytest.approx(2 * 46 / 512)


def test_mcnemar_no_discordance_and_non_binary():
    assert mcnemar_exact([1, 0, 1], [1, 0, 1]).p_value == 1.0
    assert mcnemar_exact([1, 0], [0, 1]).p_value == 1.0
    with pytest.raises(ValueError):
        mcnemar_exact([0.5, 1], [0, 1])
    with pytest.raises(ValueError):
        mcnemar_exact([0, 1, 1], [0, 1])


def test_sign_flip_detects_consistent_shift():
    rng = np.random.default_rng(0)
    d = rng.normal(0.5, 1.0, size=60)
    assert sign_flip_test(d, n_perm=2000) < 0.01
    null = rng.normal(0.0, 1.0, size=60)
    assert sign_flip_test(null, n_perm=2000) > 0.05


def test_paired_comparison_binary_runs_mcnemar():
    rng = np.random.default_rng(4)
    a = (rng.random(150) < 0.5).astype(float)
    b = a.copy()
    b[:15] = 1 - b[:15]
    res = paired_comparison(a, b, n_boot=200, n_perm=500)
    assert res.n == 150
    assert res.mcnemar is not None
    assert res.mcnemar.discordant == 15
    assert res.difference.low <= res.difference.estimate <= res.difference.high


def test_paired_comparison_real_scores_skips_mcnemar():
    rng = np.random.default_rng(5)
    a = rng.random(50)
    b = a + 0.1
    res = paired_comparison(a, b, method="percentile", n_boot=200, n_perm=500)
    assert res.mcnemar is None
    assert res.difference.estimate == pytest.approx(-0.1)
    assert res.difference.high < 0
    with pytest.raises(ValueError):
        paired_comparison(a, b[:10])


def test_pass_at_k_item_closed_form():
    from math import comb

    for n, c, k in [(10, 3, 1), (10, 3, 5), (5, 0, 2), (8, 8, 3), (6, 4, 3)]:
        expected = 1 - comb(n - c, k) / comb(n, k)
        assert pass_at_k_item(n, c, k) == pytest.approx(expected)
    with pytest.raises(ValueError):
        pass_at_k_item(3, 1, 4)


def test_pass_at_k_interval_over_items():
    rng = np.random.default_rng(6)
    n = [10] * 40
    c = rng.integers(0, 11, size=40)
    iv = pass_at_k(n, c, 1, n_boot=300)
    assert iv.estimate == pytest.approx(np.mean(np.asarray(c) / 10))
    assert iv.low <= iv.estimate <= iv.high
    with pytest.raises(ValueError):
        pass_at_k([10, 10], [1], 1)


def test_cluster_sign_flip_hand_worked():
    # three clusters with sums +3, +1, -1: total 3.
    # sign patterns give |total| in {3, 1, 5, 3, 3, 5, 1, 3}; six of eight are >= 3.
    d = [1, 1, 1, 1, -1]
    clusters = ["x", "x", "x", "y", "z"]
    assert cluster_sign_flip_test(d, clusters) == pytest.approx(6 / 8)
    # all clusters favour one side: only the identity and its mirror reach the total
    assert cluster_sign_flip_test([1, 1, 1, 1], ["a", "b", "c", "d"]) == pytest.approx(2 / 16)
    with pytest.raises(ValueError):
        cluster_sign_flip_test([1, 0], ["a"])


def test_cluster_sign_flip_random_branch_matches_exact():
    rng = np.random.default_rng(8)
    clusters = np.repeat(np.arange(14), 5)
    d = rng.normal(0.2, 1, clusters.size)
    exact = cluster_sign_flip_test(d, clusters)
    approx = cluster_sign_flip_test(d, clusters, max_exact=4, n_perm=20000, seed=1)
    assert approx == pytest.approx(exact, abs=0.015)


def test_cluster_sign_flip_size_under_null():
    """Seeded simulation: with 10 clusters and a cluster effect, the item-level
    sign-flip test rejects far too often at 5 %; the cluster test does not."""
    rng = np.random.default_rng(9)
    clusters = np.repeat(np.arange(10), 12)
    reps, rej_item, rej_cl = 200, 0, 0
    for r in range(reps):
        d = rng.normal(0, 1, 10)[clusters] + rng.normal(0, 0.5, clusters.size)
        rej_item += sign_flip_test(d, n_perm=500, seed=r) < 0.05
        rej_cl += cluster_sign_flip_test(d, clusters) < 0.05
    assert rej_item / reps > 0.3
    assert rej_cl / reps < 0.09


def test_paired_comparison_with_clusters_reports_cluster_test():
    rng = np.random.default_rng(10)
    a = (rng.random(80) < 0.6).astype(float)
    b = (rng.random(80) < 0.5).astype(float)
    clusters = np.repeat(["r1", "r2", "r3", "r4"], 20)
    res = paired_comparison(a, b, clusters=clusters, n_boot=100, n_perm=200)
    assert res.n_clusters == 4
    assert res.cluster_permutation_p is not None
    assert res.cluster_permutation_p >= 2 / 16
    assert res.to_dict()["n_clusters"] == 4


def test_paired_interval_coverage_simulation():
    """Seeded simulation: two correlated 0/1 models with a true 4-point gap.

    Each item has a latent difficulty shared by both models; the nominal 90 %
    paired CLT interval should cover the true difference about 90 % of the
    time, and be much narrower than the unpaired interval would be.
    """
    rng = np.random.default_rng(12)
    n, reps, hits = 300, 400, 0
    width = []
    for _ in range(reps):
        u = rng.random(n)
        a = (u < 0.64).astype(float)  # P(a) = 0.64
        b = (u < 0.60).astype(float)  # P(b) = 0.60, nested in a
        noise = rng.random(n) < 0.1  # independent re-draws on 10 % of items
        a[noise] = rng.random(noise.sum()) < 0.64
        res = paired_comparison(a, b, level=0.90, n_boot=10, n_perm=10)
        d = res.difference
        width.append(d.high - d.low)
        # true difference: 0.9 * 0.04 + 0.1 * (0.64 - 0.60) = 0.04
        hits += d.low <= 0.04 <= d.high
    assert abs(hits / reps - 0.90) < 0.05
    assert np.mean(width) < 0.1


def test_mcnemar_size_under_null():
    """Seeded simulation: exchangeable models, 5 % test, rejection rate <= 5 %."""
    rng = np.random.default_rng(13)
    reps, rejections = 1000, 0
    for _ in range(reps):
        u = rng.random(200)
        a = (u < 0.5).astype(float)
        b = a.copy()
        swap = rng.random(200) < 0.15
        b[swap] = rng.random(swap.sum()) < 0.5
        a[swap] = rng.random(swap.sum()) < 0.5
        rejections += mcnemar_exact(a, b).p_value < 0.05
    assert rejections / reps <= 0.06
