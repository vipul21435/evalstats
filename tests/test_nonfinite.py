import math

import numpy as np
import pytest

from evalstats import (
    cluster_sign_flip_test,
    mean_interval,
    paired_comparison,
    plan_from_scores,
    sign_flip_test,
    standard_error,
)
from evalstats.cli import main

BAD = [float("nan"), float("inf"), -float("inf")]


@pytest.mark.parametrize("bad", BAD)
def test_tests_and_intervals_reject_non_finite_scores(bad):
    x = [1.0, bad, 0.0, 1.0]
    with pytest.raises(ValueError, match="finite"):
        sign_flip_test(x)
    with pytest.raises(ValueError, match="finite"):
        cluster_sign_flip_test(x, [1, 2, 3, 4])
    for method in ("clt", "percentile", "bca"):
        with pytest.raises(ValueError, match="finite"):
            mean_interval(x, method=method)
    with pytest.raises(ValueError, match="finite"):
        standard_error(x)
    with pytest.raises(ValueError, match="finite"):
        paired_comparison(x, [0.0, 0.0, 0.0, 1.0])
    with pytest.raises(ValueError, match="finite"):
        plan_from_scores(x, [0.0, 0.0, 0.0, 1.0], 0.03)


@pytest.mark.parametrize("cell", ["nan", "inf", "-Infinity"])
def test_cli_rejects_non_finite_csv_cell(tmp_path, capsys, cell):
    p = tmp_path / "r.csv"
    p.write_text(
        f"item,model,score\nq1,a,1\nq2,a,{cell}\nq3,a,0\nq4,a,1\n"
        "q1,b,0\nq2,b,0\nq3,b,0\nq4,b,1\n",
        encoding="utf-8",
    )
    for cmd in (["compare", str(p), "--a", "a", "--b", "b"], ["leaderboard", str(p)]):
        assert main(cmd) == 2
        err = capsys.readouterr().err
        assert f"{p}:3: column 'score' is not a finite number" in err


def test_cli_rejects_json_nan(tmp_path, capsys):
    p = tmp_path / "r.jsonl"
    p.write_text('{"item": "q1", "model": "a", "score": NaN}\n', encoding="utf-8")
    assert main(["summary", str(p)]) == 2
    assert "not a finite number" in capsys.readouterr().err


def test_finite_scores_still_pass():
    x = np.array([1.0, 0.0, 1.0, 1.0])
    assert 0 < sign_flip_test(x) <= 1
    assert not math.isnan(mean_interval(x).low)
