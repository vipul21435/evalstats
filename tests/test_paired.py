import numpy as np
import pytest

from evalstats.paired import mcnemar_exact, paired_comparison, sign_flip_test
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
