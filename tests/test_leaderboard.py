import numpy as np
import pytest

from evalstats.io import ModelScores
from evalstats.leaderboard import benjamini_hochberg, holm, leaderboard


def test_holm_hand_worked():
    # p = [0.01, 0.04, 0.03]: sorted 0.01*3, 0.03*2, 0.04*1 -> 0.03, 0.06, 0.06 (monotone)
    assert holm([0.01, 0.04, 0.03]).tolist() == pytest.approx([0.03, 0.06, 0.06])
    assert holm([0.5, 0.9]).tolist() == pytest.approx([1.0, 1.0])  # lifted to the running max
    assert holm([]).size == 0


def test_bh_hand_worked():
    # p = [0.01, 0.04, 0.03]: sorted 0.01*3/1, 0.03*3/2, 0.04*3/3 -> 0.03, 0.045, 0.04
    # running minimum from the top -> 0.03, 0.04, 0.04
    assert benjamini_hochberg([0.01, 0.04, 0.03]).tolist() == pytest.approx([0.03, 0.04, 0.04])
    assert benjamini_hochberg([]).size == 0


def test_corrections_bounds_and_order():
    rng = np.random.default_rng(1)
    p = rng.random(50)
    h, q = holm(p), benjamini_hochberg(p)
    assert np.all(h >= p - 1e-15) and np.all(q >= p - 1e-15)
    assert np.all(h >= q - 1e-15) and np.all(h <= 1) and np.all(q <= 1)
    # adjusted values keep the order of the raw ones
    o = np.argsort(p)
    assert np.all(np.diff(h[o]) >= -1e-15) and np.all(np.diff(q[o]) >= -1e-15)


def _models(seed=0, n=150, real=False):
    rng = np.random.default_rng(seed)
    items = [f"c{i % 5}__{i:03d}" for i in range(n)]
    clusters = np.array([i.split("__")[0] for i in items])
    base = rng.random(n)
    out = {}
    for name, p in [("x", 0.5), ("y", 0.5), ("z", 0.7)]:
        if real:
            s = np.clip(base + rng.normal(p - 0.5, 0.2, n), 0, 1)
        else:
            s = (base < p).astype(float)
            s = np.where(rng.random(n) < 0.1, 1 - s, s)
        out[name] = ModelScores(name, items, s, clusters)
    return out


def test_leaderboard_binary_ranks_and_corrects():
    lb = leaderboard(_models(), n_boot=200)
    names = [n for n, _ in lb.models]
    assert names[0] == "z" and set(names) == {"x", "y", "z"}
    assert len(lb.pairs) == 3 and all(r.test == "mcnemar" for r in lb.pairs)
    raw = [r.p_value for r in lb.pairs]
    assert [r.p_holm for r in lb.pairs] == pytest.approx(holm(raw).tolist())
    assert [r.p_bh for r in lb.pairs] == pytest.approx(benjamini_hochberg(raw).tolist())
    # z is clearly ahead of x and y; x and y are not separated
    assert set(lb.top_group()) == {"z"}
    d = lb.to_dict()
    assert d["top_group"] == ["z"] and len(d["pairs"]) == 3


def test_leaderboard_against_leader_and_real_scores():
    lb = leaderboard(_models(real=True), against_leader=True, n_boot=200, n_perm=500)
    assert len(lb.pairs) == 2 and all(r.test == "sign-flip" for r in lb.pairs)
    assert all(r.a == lb.models[0][0] for r in lb.pairs)


def test_leaderboard_empty():
    lb = leaderboard({})
    assert lb.models == [] and lb.pairs == [] and lb.top_group() == []


def test_leaderboard_pairs_carry_paired_interval():
    lb = leaderboard(_models(), n_boot=200)
    for r in lb.pairs:
        assert r.low <= r.difference <= r.high
    z_x = next(r for r in lb.pairs if {r.a, r.b} == {"z", "x"})
    assert z_x.low > 0  # z is ranked first, so a - b is positive


def test_leaderboard_test_choices():
    models = _models()
    cl = leaderboard(models, test="cluster", n_boot=100)
    assert all(r.test == "cluster" for r in cl.pairs)
    # five clusters: no cluster p-value can go below 2 / 2**5
    assert min(r.p_value for r in cl.pairs) >= 2 / 32 - 1e-12
    sf = leaderboard(models, test="sign-flip", n_boot=100, n_perm=300)
    assert all(r.test == "sign-flip" for r in sf.pairs)
    with pytest.raises(ValueError):
        leaderboard(models, test="nope")
    no_clusters = {k: ModelScores(k, v.items, v.scores, None) for k, v in models.items()}
    with pytest.raises(ValueError):
        leaderboard(no_clusters, test="cluster", n_boot=100)
