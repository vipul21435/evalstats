import json

from evalstats.report import emit, markdown_table, pct, to_json


def test_markdown_table_alignment_and_cells():
    t = markdown_table(["name", "n", "p"], [["a", 3, 0.5], ["b", 10, None]])
    lines = t.splitlines()
    assert lines[0] == "| name | n | p |"
    assert lines[1] == "|---|---:|---|"  # p column has a None, so not numeric
    assert lines[2] == "| a | 3 | 0.5000 |"
    assert lines[3] == "| b | 10 | - |"
    assert markdown_table(["x"], []).splitlines()[1] == "|---|"


def test_emit_and_json(tmp_path, capsys):
    emit("hello")
    assert capsys.readouterr().out == "hello\n"
    out = tmp_path / "r.md"
    emit("table", out)
    assert out.read_text() == "table\n"
    assert json.loads(to_json({"a": [1, 2]})) == {"a": [1, 2]}
    assert pct(0.1234) == "12.3"
