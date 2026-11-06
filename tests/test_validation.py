import math

import pytest

from evalstats import (
    benjamini_hochberg,
    cluster_sign_flip_test,
    holm,
    mean_interval,
    pass_at_k,
    standard_error,
)


@pytest.mark.parametrize("adjust", [holm, benjamini_hochberg])
@pytest.mark.parametrize("bad", [[2.0, 0.1], [-1.0, 0.5], [0.01, math.nan]])
def test_corrections_reject_p_values_outside_unit_interval(adjust, bad):
    with pytest.raises(ValueError, match=r"p-values must be in \[0, 1\]"):
        adjust(bad)


@pytest.mark.parametrize("method", ["clt", "percentile", "bca"])
@pytest.mark.parametrize("level", [95, 0.0, 1.0, -0.5])
def test_mean_interval_rejects_level_outside_unit_interval(method, level):
    with pytest.raises(ValueError, match=r"level must be in \(0, 1\)"):
        mean_interval([1.0, 0.0, 1.0], level=level, method=method)


def test_pass_at_k_rejects_bad_level():
    with pytest.raises(ValueError, match=r"level must be in \(0, 1\)"):
        pass_at_k([3, 3], [1, 2], 1, level=95)


def test_unknown_method_is_rejected():
    with pytest.raises(ValueError, match="unknown method"):
        mean_interval([1.0, 0.0], method="wald")  # type: ignore[arg-type]


def test_cluster_keys_may_include_none():
    d = [1.0, 0.0, -1.0, 1.0, 1.0, 0.0]
    keys = ["r1", None, "r2", None, "r1", 3]
    p = cluster_sign_flip_test(d, keys)
    assert 0 < p <= 1
    assert standard_error(d, keys) > 0
    iv = mean_interval(d, method="bca", clusters=keys, n_boot=200)
    assert iv.n_clusters == 4  # None is a key of its own


def _write_small(tmp_path):
    import json

    rows = []
    for i in range(40):
        rows.append({"item": f"q{i:02d}", "model": "A", "score": int(i < 20)})
        rows.append({"item": f"q{i:02d}", "model": "B", "score": int(0 < i < 20)})
    path = tmp_path / "small.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return str(path)


def test_power_error_reports_observed_discordance(tmp_path, capsys):
    from evalstats.cli import main

    path = _write_small(tmp_path)
    assert main(["power", path, "--a", "A", "--b", "B", "--delta", "0.03"]) == 2
    err = capsys.readouterr().err
    assert "observed discordance 0.025" in err and "--discordance" in err
    assert main(["power", path, "--a", "A", "--b", "B", "--delta", "0.02"]) == 0
    assert "more than the observed discordance" in capsys.readouterr().out


def test_compare_refuses_the_same_model_twice(tmp_path, capsys):
    from evalstats.cli import main

    path = _write_small(tmp_path)
    for cmd in (["compare"], ["power", "--delta", "0.01"]):
        assert main([cmd[0], path, "--a", "A", "--b", "A", *cmd[1:]]) == 2
        assert "same model 'A'" in capsys.readouterr().err


def test_passk_error_names_model_item_and_samples(tmp_path, capsys):
    from evalstats.cli import main

    path = tmp_path / "samples.csv"
    path.write_text("item,model,n_samples,n_correct\nq1,m,10,3\nq2,m,10,4\n")
    assert main(["passk", str(path), "--k", "11"]) == 2
    err = capsys.readouterr().err
    assert "'m'" in err and "'q1'" in err and "10 samples" in err and "k = 11" in err
