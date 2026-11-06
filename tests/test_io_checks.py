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
