import json

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
