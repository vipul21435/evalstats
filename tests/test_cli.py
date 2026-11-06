import csv
import json

import numpy as np
import pytest

from evalstats.cli import main
from evalstats.io import align, read_records, scores_by_model


def _write_fixture(path, fmt="jsonl"):
    rng = np.random.default_rng(10)
    rows = []
    for i in range(40):
        item = f"src{i % 4}__{i:03d}"
        a = int(rng.random() < 0.5)
        b = a if rng.random() < 0.8 else 1 - a
        rows.append({"item": item, "model": "alpha", "score": a})
        rows.append({"item": item, "model": "beta", "score": b})
    if fmt == "csv":
        with path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["item", "model", "score"])
            w.writeheader()
            w.writerows(rows)
    else:
        path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return rows


@pytest.mark.parametrize("fmt", ["jsonl", "csv"])
def test_read_records_both_formats(tmp_path, fmt):
    p = tmp_path / f"r.{fmt}"
    rows = _write_fixture(p, fmt)
    recs = read_records(p)
    assert len(recs) == len(rows)
    assert recs[0].item == "src0__000" and recs[0].cluster is None
    models = scores_by_model(recs, default_cluster=True)
    assert set(models) == {"alpha", "beta"}
    assert sorted(set(models["alpha"].clusters)) == ["src0", "src1", "src2", "src3"]


def test_align_restricts_to_shared_items():
    recs = [
        *(dict(item=f"i{k}", model="x", score=1) for k in range(5)),
        *(dict(item=f"i{k}", model="y", score=0) for k in range(2, 8)),
    ]
    from evalstats.io import Record

    models = scores_by_model([Record(**r) for r in recs])
    shared, sa, sb, cl = align(models["x"], models["y"])
    assert shared == ["i2", "i3", "i4"]
    assert sa.tolist() == [1, 1, 1] and sb.tolist() == [0, 0, 0] and cl is None
    with pytest.raises(ValueError):
        align(models["x"], scores_by_model([Record("z", "w", 1.0)])["w"])


def test_summary_markdown_and_json(tmp_path, capsys):
    p = tmp_path / "r.jsonl"
    _write_fixture(p)
    assert main(["summary", str(p), "--cluster-by-prefix"]) == 0
    out = capsys.readouterr().out
    assert "| alpha | 40 | 4 |" in out
    assert main(["summary", str(p), "--json", "--method", "bca", "--n-boot", "200"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert [d["model"] for d in data] == ["alpha", "beta"]
    assert data[0]["method"] == "bca"


def test_compare_markdown_and_json(tmp_path, capsys):
    p = tmp_path / "r.csv"
    _write_fixture(p, "csv")
    args = ["compare", str(p), "--a", "alpha", "--b", "beta", "--n-perm", "300"]
    assert main(args) == 0
    out = capsys.readouterr().out
    assert "shared items: 40" in out and "McNemar exact" in out
    assert main([*args, "--json", "--method", "percentile", "--n-boot", "200"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["n"] == 40 and data["mcnemar"]["discordant"] > 0
    assert main(["compare", str(p), "--a", "alpha", "--b", "nope"]) == 2


def test_passk_command(tmp_path, capsys):
    p = tmp_path / "k.jsonl"
    rows = [
        {"item": f"i{j}", "model": "m", "n_samples": 10, "n_correct": j % 4} for j in range(30)
    ]
    p.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    assert main(["passk", str(p), "--k", "2", "--n-boot", "200"]) == 0
    out = capsys.readouterr().out
    assert "pass@2" in out and "| m | 30 |" in out
    assert main(["passk", str(p), "--json", "--n-boot", "100"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data[0]["k"] == 1 and data[0]["n"] == 30
    empty = tmp_path / "e.jsonl"
    empty.write_text(json.dumps({"item": "a", "model": "m"}) + "\n")
    assert main(["passk", str(empty)]) == 2


def test_passk_per_sample_rows(tmp_path, capsys):
    p = tmp_path / "s.jsonl"
    rows = []
    for j in range(12):
        for s in range(5):
            rows.append({"item": f"i{j}", "model": "m", "sample": s, "score": int(s < j % 3)})
    p.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    assert main(["passk", str(p), "--k", "1", "--json", "--n-boot", "100"]) == 0
    data = json.loads(capsys.readouterr().out)
    # each item has n=5 samples and c = j % 3 correct; pass@1 = mean(c / 5)
    expected = sum((j % 3) / 5 for j in range(12)) / 12
    assert data[0]["n"] == 12 and abs(data[0]["estimate"] - expected) < 1e-12
    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps({"item": "a", "model": "m", "score": 0.5}) + "\n")
    with pytest.raises(ValueError):
        main(["passk", str(bad)])
