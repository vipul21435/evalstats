import numpy as np

from evalstats import mean_interval, paired_comparison
from evalstats.cli import main
from evalstats.io import ModelScores
from evalstats.leaderboard import leaderboard


def test_clt_bounds_are_clipped_to_the_given_range():
    iv = mean_interval([1.0, 0.0, 0.0, 0.0], bounds=(0.0, 1.0))
    assert iv.low == 0.0 and iv.high < 1.0 and iv.clipped
    plain = mean_interval([1.0, 0.0, 0.0, 0.0])
    assert plain.low < 0 and not plain.clipped
    inside = mean_interval([1.0, 0.0] * 50, bounds=(0.0, 1.0))
    assert not inside.clipped and inside.low > 0


def test_paired_difference_of_binary_scores_stays_within_one():
    res = paired_comparison([1.0, 0.0], [0.0, 1.0])
    assert -1.0 <= res.difference.low and res.difference.high <= 1.0
    assert res.difference.clipped
    real = paired_comparison([0.3, 2.5], [0.1, -2.0])
    assert not real.difference.clipped


def test_leaderboard_clips_model_and_pair_intervals():
    items = ["i1", "i2", "i3", "i4"]
    models = {
        "x": ModelScores("x", items, np.array([1.0, 0.0, 0.0, 0.0]), None),
        "y": ModelScores("y", items, np.array([0.0, 1.0, 1.0, 1.0]), None),
    }
    lb = leaderboard(models, n_perm=100)
    for _, iv in lb.models:
        assert 0.0 <= iv.low and iv.high <= 1.0
    (pair,) = lb.pairs
    assert -1.0 <= pair.low and pair.high <= 1.0


def test_cli_notes_clipped_bounds(tmp_path, capsys):
    p = tmp_path / "small.csv"
    p.write_text("item,model,score\nq1,a,1\nq2,a,0\nq3,a,0\nq4,a,0\n", encoding="utf-8")
    assert main(["summary", str(p)]) == 0
    out = capsys.readouterr().out
    assert "| a | 4 | - | 25.0 | 0.0 |" in out
    assert "clipped" in out and "--method bca" in out
    q = tmp_path / "two.csv"
    q.write_text("item,model,score\nq1,a,1\nq2,a,0\nq1,b,0\nq2,b,1\n", encoding="utf-8")
    assert main(["compare", str(q), "--a", "a", "--b", "b"]) == 0
    out = capsys.readouterr().out
    assert "[-100.0, 100.0]" in out and "clipped" in out


def test_missing_bounds_are_not_marked_clipped(tmp_path, capsys):
    for method in ("clt", "percentile", "bca"):
        iv = mean_interval([1.0], method=method, bounds=(0.0, 1.0))
        assert np.isnan(iv.low) and not iv.clipped
        one_cluster = mean_interval([1.0, 0.0], method=method, clusters=["r", "r"], bounds=(0, 1))
        assert not one_cluster.clipped
    path = tmp_path / "one.csv"
    path.write_text("item,model,score\nq1,m,1\n")
    assert main(["summary", str(path), "--method", "bca"]) == 0
    out = capsys.readouterr().out
    assert "fewer than 2 items" in out and "clipped" not in out
