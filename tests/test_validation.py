import math

import pytest

from evalstats import (
    benjamini_hochberg,
    cluster_sign_flip_test,
    holm,
    mean_interval,
    pass_at_k,
    standard_error,
)


@pytest.mark.parametrize("adjust", [holm, benjamini_hochberg])
@pytest.mark.parametrize("bad", [[2.0, 0.1], [-1.0, 0.5], [0.01, math.nan]])
def test_corrections_reject_p_values_outside_unit_interval(adjust, bad):
    with pytest.raises(ValueError, match=r"p-values must be in \[0, 1\]"):
        adjust(bad)


@pytest.mark.parametrize("method", ["clt", "percentile", "bca"])
@pytest.mark.parametrize("level", [95, 0.0, 1.0, -0.5])
def test_mean_interval_rejects_level_outside_unit_interval(method, level):
    with pytest.raises(ValueError, match=r"level must be in \(0, 1\)"):
        mean_interval([1.0, 0.0, 1.0], level=level, method=method)


def test_pass_at_k_rejects_bad_level():
    with pytest.raises(ValueError, match=r"level must be in \(0, 1\)"):
        pass_at_k([3, 3], [1, 2], 1, level=95)


def test_unknown_method_is_rejected():
    with pytest.raises(ValueError, match="unknown method"):
        mean_interval([1.0, 0.0], method="wald")  # type: ignore[arg-type]


def test_cluster_keys_may_include_none():
    d = [1.0, 0.0, -1.0, 1.0, 1.0, 0.0]
    keys = ["r1", None, "r2", None, "r1", 3]
    p = cluster_sign_flip_test(d, keys)
    assert 0 < p <= 1
    assert standard_error(d, keys) > 0
    iv = mean_interval(d, method="bca", clusters=keys, n_boot=200)
    assert iv.n_clusters == 4  # None is a key of its own
