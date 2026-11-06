import json

import pytest

from evalstats.cli import main
from evalstats.io import Record, passk_counts, read_records


def test_csv_with_byte_order_mark(tmp_path):
    p = tmp_path / "bom.csv"
    p.write_bytes(b"\xef\xbb\xbfitem,model,score\nq1,a,1\n")
    (r,) = read_records(p)
    assert (r.item, r.model, r.score) == ("q1", "a", 1.0)


def test_non_utf8_file_names_the_file(tmp_path, capsys):
    p = tmp_path / "l1.csv"
    p.write_bytes(b"item,model,score\nq\xe91,a,1\n")
    with pytest.raises(ValueError, match="l1.csv: not UTF-8 text"):
        read_records(p)
    assert main(["summary", str(p)]) == 2
    assert "l1.csv: not UTF-8 text" in capsys.readouterr().err


@pytest.mark.parametrize("cmd", ["summary", "leaderboard", "report"])
def test_file_without_scored_rows_is_an_error(tmp_path, capsys, cmd):
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    counts = tmp_path / "counts.jsonl"
    counts.write_text('{"item": "q1", "model": "a", "n_samples": 3, "n_correct": 1}\n')
    for p in (empty, counts):
        assert main([cmd, str(p)]) == 2
        assert f"no scored rows in {p}" in capsys.readouterr().err


def test_passk_merges_direct_and_per_sample_items():
    recs = [
        Record("q1", "a", None, n_samples=3, n_correct=1),
        Record("q2", "a", 1.0),
        Record("q2", "a", 0.0),
    ]
    assert passk_counts(recs) == {"a": [(3, 1), (2, 1)]}


def test_passk_refuses_both_kinds_of_row_for_one_item():
    recs = [Record("q1", "a", None, n_samples=3, n_correct=1), Record("q1", "a", 1.0)]
    with pytest.raises(ValueError, match="q1"):
        passk_counts(recs)


def test_passk_refuses_repeated_count_rows_for_one_item():
    recs = [
        Record("q1", "a", None, n_samples=3, n_correct=1),
        Record("q1", "a", None, n_samples=3, n_correct=2),
    ]
    with pytest.raises(ValueError, match="q1"):
        passk_counts(recs)


@pytest.mark.parametrize("value", ["3.7", '"3.5"', "true"])
def test_fractional_counts_are_rejected(tmp_path, value):
    p = tmp_path / "pk.jsonl"
    p.write_text(f'{{"item": "q1", "model": "a", "n_samples": {value}, "n_correct": 1}}\n')
    with pytest.raises(ValueError, match="pk.jsonl:1: column 'n_samples' is not a whole number"):
        read_records(p)


def test_whole_float_counts_are_accepted(tmp_path):
    p = tmp_path / "pk.jsonl"
    p.write_text('{"item": "q1", "model": "a", "n_samples": 3.0, "n_correct": 1}\n')
    (r,) = read_records(p)
    assert r.n_samples == 3 and isinstance(r.n_samples, int)


def test_one_count_without_the_other_is_rejected(tmp_path):
    p = tmp_path / "pk.jsonl"
    p.write_text('{"item": "q1", "model": "a", "n_samples": 3}\n')
    with pytest.raises(ValueError, match="pk.jsonl:1: n_samples and n_correct go together"):
        read_records(p)


@pytest.mark.parametrize("line", ["q1,a", "q1,a,1,extra"])
def test_csv_row_with_wrong_field_count(tmp_path, line):
    p = tmp_path / "short.csv"
    p.write_text(f"item,model,score\nq0,a,1\n{line}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="short.csv:3: expected 3 fields"):
        read_records(p)


def test_json_scores_survive_round_trip(tmp_path):
    p = tmp_path / "r.jsonl"
    p.write_text(json.dumps({"item": 1, "model": "a", "score": 0.5}) + "\n")
    (r,) = read_records(p)
    assert r.item == "1" and r.score == 0.5


def test_blank_cluster_beside_filled_ones_is_an_error(tmp_path, capsys):
    from evalstats.io import scores_by_model

    rows = ["item,model,score,cluster"]
    rows += [f"q{i:02d},m,{i % 2},r{i % 10}" for i in range(40)]
    rows += [f"u{i:02d},m,{i % 2}," for i in range(20)]
    path = tmp_path / "partial.csv"
    path.write_text("\n".join(rows) + "\n")
    with pytest.raises(ValueError, match="'u00' has no cluster"):
        scores_by_model(read_records(path))
    assert main(["summary", str(path)]) == 2
    assert "u00" in capsys.readouterr().err


def test_cluster_from_a_later_sample_row_counts():
    from evalstats.io import scores_by_model

    recs = [Record("q1", "m", 1.0), Record("q1", "m", 0.0, "r1"), Record("q2", "m", 1.0, "r2")]
    assert scores_by_model(recs)["m"].clusters.tolist() == ["r1", "r2"]


def test_align_takes_clusters_from_either_model_and_checks_them():
    import numpy as np

    from evalstats.io import ModelScores, align

    items = ["x1", "x2", "y1"]
    with_keys = ModelScores("a", items, np.array([1.0, 0.0, 1.0]), np.array(["x", "x", "y"]))
    without = ModelScores("b", items, np.array([0.0, 0.0, 1.0]), None)
    assert align(without, with_keys)[3].tolist() == ["x", "x", "y"]
    assert align(with_keys, without)[3].tolist() == ["x", "x", "y"]
    other = ModelScores("c", items, np.zeros(3), np.array(["x", "z", "y"]))
    with pytest.raises(ValueError, match="'x2' is in cluster 'x' for model 'a'"):
        align(with_keys, other)


def test_prefix_cluster_splits_at_the_double_underscore():
    from evalstats.io import scores_by_model

    recs = [Record("my_repo__issue-1", "m", 1.0), Record("other__2", "m", 0.0)]
    got = scores_by_model(recs, default_cluster=True)["m"].clusters.tolist()
    assert got == ["my_repo", "other"]
