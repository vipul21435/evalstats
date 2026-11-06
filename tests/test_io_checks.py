
import pytest

from evalstats.cli import main
from evalstats.io import read_records


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
