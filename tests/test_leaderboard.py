import numpy as np
import pytest

from evalstats.intervals import Interval
from evalstats.io import ModelScores
from evalstats.leaderboard import (
    Leaderboard,
    PairRow,
    benjamini_hochberg,
    holm,
    leaderboard,
)


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


def _subset_full():
    # "full" solves the first 100 items and a third of 200 more (mean 55.3 %);
    # "subset" ran only the first 100 and failed 12 of them (mean 88 %). On the
    # shared items full is ahead, significantly, but subset leads on its mean.
    items = [f"q{i:03d}" for i in range(300)]
    full = np.array([1.0] * 100 + [1.0 if i % 3 == 0 else 0.0 for i in range(200)])
    sub = np.array([0.0] * 12 + [1.0] * 88)
    return {
        "full": ModelScores("full", items, full, None),
        "subset": ModelScores("subset", items[:100], sub, None),
    }


def test_top_group_keeps_a_model_that_beats_the_leader_on_shared_items():
    lb = leaderboard(_subset_full())
    assert lb.models[0][0] == "subset"
    (row,) = lb.pairs
    assert row.p_holm < lb.alpha
    # the significant pair favours "full", so it is not beaten by the leader
    assert sorted(lb.top_group()) == ["full", "subset"]


def test_leaderboard_notes_differing_item_sets():
    from evalstats.report import leaderboard_tables

    text = leaderboard_tables(leaderboard(_subset_full()))
    assert "different item sets" in text
    assert "different item sets" not in leaderboard_tables(leaderboard(_models()))


def _row(a, b, diff, p, p_holm, p_bh):
    return PairRow(a, b, 100, diff, diff - 0.05, diff + 0.05, "mcnemar", p, p_holm, p_bh)


def _iv(est):
    return Interval(est, est - 0.1, est + 0.1, 0.95, "clt", 100, 0.05, None)


def test_top_group_uses_holm_not_bh():
    # raw p 0.03 and 0.04: Holm gives 0.06 for both, BH gives 0.04 for both
    lb = Leaderboard(
        [("l", _iv(0.6)), ("m", _iv(0.5)), ("n", _iv(0.4))],
        [_row("l", "m", 0.1, 0.03, 0.06, 0.04), _row("l", "n", 0.2, 0.04, 0.06, 0.04)],
        0.05,
    )
    assert lb.top_group() == ["l", "m", "n"]
    # at alpha 0.1 both pairs separate
    lb10 = Leaderboard(lb.models, lb.pairs, 0.1)
    assert lb10.top_group() == ["l"]


def test_leaderboard_alpha_follows_level():
    assert leaderboard(_models(), n_boot=50).alpha == 0.05
    lb = leaderboard(_models(), level=0.9, n_boot=50)
    assert lb.alpha == 0.1 and lb.to_dict()["alpha"] == 0.1
