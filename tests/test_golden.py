"""Golden outputs of the quickstart commands on the committed example file."""

from pathlib import Path

import pytest

from evalstats.cli import main

RESULTS = str(Path(__file__).resolve().parents[1] / "examples" / "results.jsonl")


def _run(capsys, *argv):
    assert main(list(argv)) == 0
    return capsys.readouterr().out


def test_quickstart_compare_lines(capsys):
    out = _run(capsys, "compare", RESULTS, "--a", "a", "--b", "b")
    assert "difference (a - b): -5.0 points, 95% interval [-8.9, -1.1] (clt)" in out
    assert "sign-flip permutation p: 0.0332" in out
    assert "a-only 0, b-only 6, discordant 6, p = 0.0312" in out
    out = _run(capsys, "compare", RESULTS, "--a", "a", "--b", "b", "--cluster-by-prefix")
    assert "95% interval [-10.3, 0.3] (clt)" in out
    assert "cluster sign-flip p (8 clusters): 0.1250" in out
    out = _run(capsys, "compare", RESULTS, "--a", "a", "--b", "c")
    assert "sign-flip permutation p: 0.6860" in out
    assert "a-only 25, c-only 29, discordant 54, p = 0.6835" in out


def test_quickstart_summary_bca(capsys):
    out = _run(capsys, "summary", RESULTS, "--cluster-by-prefix", "--method", "bca")
    assert "| a | 120 | 8 | 43.3 | 34.4 | 53.1 | bca |" in out
    assert "| b | 120 | 8 | 48.3 | 39.1 | 60.6 | bca |" in out
    assert "| c | 120 | 8 | 46.7 | 31.5 | 60.5 | bca |" in out


@pytest.mark.parametrize(
    "line",
    [
        "observed discordance: 5.0 % of items",
        "434 (independent items), 434 (with the design effect)",
        "power for 3.0 points on the observed 120 items: 0.32",
        "smallest difference detectable on 120 items: 5.6 points",
    ],
)
def test_quickstart_power(capsys, line):
    out = _run(capsys, "power", RESULTS, "--a", "a", "--b", "b", "--delta", "0.03")
    assert line in out
