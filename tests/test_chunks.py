import numpy as np
import pytest

import evalstats.intervals as intervals
from evalstats import cluster_sign_flip_test, mean_interval, pass_at_k, sign_flip_test
from evalstats.intervals import row_chunks


def test_row_chunks_cover_all_rows():
    assert row_chunks(10, 1 << 22) == [1] * 10
    assert sum(row_chunks(10000, 333)) == 10000
    assert row_chunks(0, 5) == []


def _results():
    rng = np.random.default_rng(3)
    x = (rng.random(137) < 0.4).astype(float)
    y = (rng.random(137) < 0.5).astype(float)
    cl = np.arange(137) % 23
    return (
        sign_flip_test(x - y, n_perm=3001),
        cluster_sign_flip_test(x - y, cl, n_perm=3001),
        mean_interval(x, method="bca", n_boot=1501).to_dict(),
        mean_interval(x, method="percentile", clusters=cl, n_boot=1501).to_dict(),
        pass_at_k([5] * 137, (x * 3).astype(int), 2, n_boot=1501).to_dict(),
    )


def test_results_do_not_depend_on_chunk_size(monkeypatch):
    whole = _results()
    # tiny chunks: a handful of rows per draw, odd sizes included
    monkeypatch.setattr(intervals, "CHUNK_ELEMENTS", 997)
    assert row_chunks(3001, 137)[0] == 7
    assert _results() == whole


@pytest.mark.parametrize("width", [1, 7, 120, 1001])
def test_chunked_draws_equal_one_draw(width):
    one = np.random.default_rng(0).integers(0, width, size=(2000, width))
    rng = np.random.default_rng(0)
    parts = np.concatenate([rng.integers(0, width, size=(m, width)) for m in (700, 700, 600)])
    assert (one == parts).all()
