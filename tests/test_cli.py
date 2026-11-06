import csv
import json

import numpy as np
import pytest

from evalstats.cli import main
from evalstats.io import Record, align, read_records, scores_by_model


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
    models = scores_by_model([Record(**r) for r in recs])
    shared, sa, sb, cl = align(models["x"], models["y"])
    assert shared == ["i2", "i3", "i4"]
    assert sa.tolist() == [1, 1, 1] and sb.tolist() == [0, 0, 0] and cl is None
    with pytest.raises(ValueError):
        align(models["x"], scores_by_model([Record("z", "w", 1.0)])["w"])


def test_repeated_rows_are_averaged_per_item():
    recs = [
        Record("i1", "m", 1.0, "r1"),
        Record("i1", "m", 0.0, "r1"),
        Record("i1", "m", 1.0, "r1"),
        Record("i2", "m", 0.0, "r2"),
        Record("i2", "m", None),
    ]
    m = scores_by_model(recs)["m"]
    assert m.items == ["i1", "i2"]
    assert m.scores.tolist() == pytest.approx([2 / 3, 0.0])
    assert m.clusters is not None and m.clusters.tolist() == ["r1", "r2"]


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
    assert "difference (alpha - beta)" in out and "alpha-only" in out and "beta-only" in out
    assert main([*args, "--json", "--method", "percentile", "--n-boot", "200"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["n"] == 40 and data["mcnemar"]["discordant"] > 0
    assert main(["compare", str(p), "--a", "alpha", "--b", "nope"]) == 2


def test_passk_command(tmp_path, capsys):
    p = tmp_path / "k.jsonl"
    rows = [{"item": f"i{j}", "model": "m", "n_samples": 10, "n_correct": j % 4} for j in range(30)]
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
    assert main(["passk", str(bad)]) == 2
    assert "0/1 scores" in capsys.readouterr().err


def test_leaderboard_and_power_commands(tmp_path, capsys):
    p = tmp_path / "r.jsonl"
    _write_fixture(p)
    assert main(["leaderboard", str(p), "--cluster-by-prefix", "--n-boot", "100"]) == 0
    out = capsys.readouterr().out
    assert "| rank | model |" in out and "p Holm" in out and "not separated" in out
    assert main(["leaderboard", str(p), "--json", "--against-leader", "--models", "beta"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert [m["model"] for m in data["models"]] == ["beta"] and data["pairs"] == []
    assert main(["leaderboard", str(p), "--models", "nope"]) == 2
    assert main(["leaderboard", str(p), "--cluster-by-prefix", "--test", "cluster"]) == 0
    assert "| cluster |" in capsys.readouterr().out
    out_md = tmp_path / "power.md"
    assert (
        main(
            [
                "power",
                str(p),
                "--a",
                "alpha",
                "--b",
                "beta",
                "--delta",
                "0.03",
                "--cluster-by-prefix",
                "--out",
                str(out_md),
            ]
        )
        == 0
    )
    text = out_md.read_text()
    assert "observed discordance" in text and "design effect" in text
    assert main(["power", str(p), "--a", "alpha", "--b", "beta", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["n_items"] > 0 and data["design_effect"] == 1.0
    assert main(["power", str(p), "--a", "alpha", "--b", "gamma"]) == 2
    assert main(["summary", str(p), "--json", "--out", str(tmp_path / "s.json")]) == 0
    assert json.loads((tmp_path / "s.json").read_text())[0]["model"] == "alpha"
    assert (
        main(["compare", str(p), "--a", "alpha", "--b", "beta", "--out", str(tmp_path / "c.md")])
        == 0
    )
    assert "McNemar" in (tmp_path / "c.md").read_text()


def test_report_command(tmp_path, capsys):
    p = tmp_path / "r.jsonl"
    _write_fixture(p)
    out = tmp_path / "report.md"
    args = ["report", str(p), "--cluster-by-prefix", "--n-boot", "100", "--out", str(out)]
    assert main(args) == 0
    text = out.read_text()
    assert "## Intervals and pairwise tests" in text and "r.jsonl" in text
    assert main(["report", str(p), "--json", "--n-boot", "100", "--test", "sign-flip"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["n_models"] >= 2 and "plans" in data


def test_power_without_data(tmp_path, capsys):
    from evalstats.power import items_for_discordance

    assert main(["power", "--discordance", "0.12", "--delta", "0.03"]) == 0
    out = capsys.readouterr().out
    assert out.strip().endswith(f": {items_for_discordance(0.03, 0.12)}")
    assert main(["power", "--discordance", "0.2", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["n_items"] == items_for_discordance(0.03, 0.2)
    assert main(["power"]) == 2
    p = tmp_path / "r.jsonl"
    _write_fixture(p)
    assert main(["power", str(p), "--a", "alpha"]) == 2


def _bad_inputs(tmp_path):
    bad_json = tmp_path / "bad.jsonl"
    bad_json.write_text('{"item": "x", "model": "a", "score": 1}\n{bad\n', encoding="utf-8")
    no_model = tmp_path / "nomodel.jsonl"
    no_model.write_text('{"item": "x", "score": 1}\n', encoding="utf-8")
    bad_score = tmp_path / "badscore.csv"
    bad_score.write_text("item,model,score\nx,a,yes\n", encoding="utf-8")
    real = tmp_path / "real.jsonl"
    real.write_text(
        "".join(
            json.dumps({"item": f"i{k}", "model": m, "score": s}) + "\n"
            for k in range(3)
            for m, s in (("a", 0.5), ("b", 0.2))
        ),
        encoding="utf-8",
    )
    pk = tmp_path / "pk.jsonl"
    pk.write_text('{"item": "x", "model": "a", "n_samples": 3, "n_correct": 1}\n')
    fixture = tmp_path / "ok.jsonl"
    _write_fixture(fixture)
    return {
        "missing": (["summary", str(tmp_path / "missing.jsonl")], "No such file"),
        "json": (["summary", str(bad_json)], "bad.jsonl:2: not valid JSON"),
        "column": (["summary", str(no_model)], "nomodel.jsonl:1: missing column 'model'"),
        "number": (["summary", str(bad_score)], "badscore.csv:2: column 'score'"),
        "power": (["power", "--discordance", "0.02", "--delta", "0.03"], "cannot exceed"),
        "passk": (["passk", str(pk), "--k", "5"], "3 samples, fewer than k = 5"),
        "cluster": (["leaderboard", str(fixture), "--test", "cluster"], "--cluster-by-prefix"),
        "mcnemar": (["leaderboard", str(real), "--test", "mcnemar"], "McNemar needs 0/1"),
    }


@pytest.mark.parametrize(
    "case", ["missing", "json", "column", "number", "power", "passk", "cluster", "mcnemar"]
)
def test_bad_input_is_one_line_error(tmp_path, capsys, case):
    argv, expected = _bad_inputs(tmp_path)[case]
    assert main(argv) == 2
    err = capsys.readouterr().err
    assert err.startswith("error: ") and expected in err
    assert "Traceback" not in err and err.count("\n") == 1


@pytest.mark.parametrize(
    "argv",
    [
        ["summary", "x.jsonl", "--level", "1.5"],
        ["power", "--discordance", "0", "--delta", "0.03"],
        ["power", "--discordance", "0.1", "--delta", "-1"],
        ["power", "--discordance", "0.1", "--alpha", "abc"],
        ["passk", "x.jsonl", "--k", "0"],
    ],
)
def test_out_of_range_flags_are_usage_errors(argv, capsys):
    with pytest.raises(SystemExit) as exc:
        main(argv)
    assert exc.value.code == 2
    assert "Traceback" not in capsys.readouterr().err


def test_summary_with_one_item_prints_dashes(tmp_path, capsys):
    p = tmp_path / "one.jsonl"
    p.write_text(
        '{"item": "x", "model": "a", "score": 1}\n'
        '{"item": "x", "model": "b", "score": 0}\n'
        '{"item": "y", "model": "b", "score": 1}\n',
        encoding="utf-8",
    )
    assert main(["summary", str(p)]) == 0
    out = capsys.readouterr().out
    assert "| a | 1 | - | 100.0 | - | - | clt |" in out
    assert "nan" not in out and "fewer than 2 items" in out


def test_help_has_no_markup(capsys):
    with pytest.raises(SystemExit):
        main(["--help"])
    assert "``" not in capsys.readouterr().out
