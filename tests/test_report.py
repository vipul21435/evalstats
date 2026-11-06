import json

import numpy as np

from evalstats.report import emit, markdown_table, p_value, pct, to_json


def test_markdown_table_alignment_and_cells():
    t = markdown_table(["name", "n", "p"], [["a", 3, 0.5], ["b", 10, None]])
    lines = t.splitlines()
    assert lines[0] == "| name | n | p |"
    assert lines[1] == "|---|---:|---:|"  # a missing value keeps a column numeric
    assert lines[2] == "| a | 3 | 0.5000 |"
    assert lines[3] == "| b | 10 | - |"
    assert markdown_table(["x"], []).splitlines()[1] == "|---|"
    assert markdown_table(["x"], [[None], ["-"]]).splitlines()[1] == "|---|"


def test_string_numbers_are_right_aligned():
    t = markdown_table(["m", "mean %", "p"], [["a", "71.8", "<0.0001"], ["b", "-3.0", "0.0627"]])
    assert t.splitlines()[1] == "|---|---:|---:|"
    assert markdown_table(["flag"], [[True]]).splitlines()[1] == "|---|"


def test_p_value_format():
    assert p_value(0.04567) == "0.0457"
    assert p_value(1.0) == "1.0000"
    assert p_value(3e-7) == "<0.0001"


def test_emit_and_json(tmp_path, capsys):
    emit("hello")
    assert capsys.readouterr().out == "hello\n"
    out = tmp_path / "r.md"
    emit("table", out)
    assert out.read_text() == "table\n"
    assert json.loads(to_json({"a": [1, 2]})) == {"a": [1, 2]}
    assert pct(0.1234) == "12.3"


def _models(n=120):
    import numpy as np

    from evalstats.io import ModelScores

    rng = np.random.default_rng(0)
    items = [f"r{i % 6}__{i:03d}" for i in range(n)]
    clusters = np.array([i.split("__")[0] for i in items])
    base = rng.random(n)
    out = {}
    for name, p in [("m1", 0.45), ("m2", 0.5), ("m3", 0.7)]:
        s = (base < p).astype(float)
        s = np.where(rng.random(n) < 0.08, 1 - s, s)
        out[name] = ModelScores(name, items, s, clusters)
    return out


def test_full_report_markdown_and_json():
    from evalstats.report import full_report, render_markdown, report_json

    rep = full_report(_models(), source="toy.jsonl", delta=0.03, n_boot=100)
    assert rep["n_models"] == 3 and rep["n_items"] == 120
    # one plan per pair of neighbours in the ranking
    assert len(rep["plans"]) == 2
    md = render_markdown(rep)
    assert md.startswith("# evalstats report")
    assert "toy.jsonl" in md and "p Holm" in md and "items (clustered)" in md
    data = json.loads(report_json(rep))
    assert data["leaderboard"]["models"][0]["model"] == "m3"
    assert data["plans"][0]["n_items"] > 0


def test_full_report_skips_pairs_with_too_little_discordance():
    from evalstats.report import full_report

    rep = full_report(_models(), source="toy.jsonl", delta=0.9, n_boot=100)
    assert rep["plans"] == []


def test_full_report_skips_pairs_that_never_disagree():
    from evalstats.io import ModelScores
    from evalstats.report import full_report, render_markdown

    items = [f"i{k}" for k in range(6)]
    same = np.array([1.0, 0.0, 1.0, 1.0, 0.0, 1.0])
    models = {
        "x": ModelScores("x", items, same, None),
        "y": ModelScores("y", items, same.copy(), None),
    }
    rep = full_report(models, source="same.jsonl", delta=0.03, n_boot=50, n_perm=50)
    assert rep["plans"] == []
    assert "agree on every item" in render_markdown(rep)


def test_cli_power_on_identical_models_is_an_error(tmp_path, capsys):
    from evalstats.cli import main

    p = tmp_path / "same.csv"
    p.write_text("item,model,score\nq1,a,1\nq2,a,0\nq3,a,1\nq1,b,1\nq2,b,0\nq3,b,1\n")
    assert main(["power", str(p), "--a", "a", "--b", "b"]) == 2
    err = capsys.readouterr().err
    assert "agree on every shared item" in err and "--discordance" in err
    assert ": 0 (independent items)" not in err
