"""Offline test of examples/fetch_swebench.py: files are pre-placed in its cache."""

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "examples" / "fetch_swebench.py"


def _load():
    spec = importlib.util.spec_from_file_location("fetch_swebench", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _place(cache: Path, sub: str, resolved: list[bool], reported: float | None) -> None:
    details = {f"own__repo-{k}": {"resolved": r} for k, r in enumerate(resolved)}
    (cache / f"{sub}.json").write_text(json.dumps(details), encoding="utf-8")
    meta = "info:\n  name: x\n"
    if reported is not None:
        meta += f"  resolved: {reported}\n"
    (cache / f"{sub}.metadata.yaml").write_text(meta, encoding="utf-8")


def test_fetch_writes_rows_and_skips_mismatched_submission(tmp_path, capsys):
    mod = _load()
    cache = tmp_path / "submissions" / "abc123"
    cache.mkdir(parents=True)
    _place(cache, "20250101_agent_good", [True, False, True, True], 75.0)
    _place(cache, "20250101_agent_broken", [False] * 4, 50.0)
    _place(cache, "nodate", [True, False], None)
    argv = ["--out", str(tmp_path), "--ref", "abc123", "--submissions"]
    argv += ["20250101_agent_good", "20250101_agent_broken", "nodate"]
    assert mod.main(argv) == 0
    rows = [json.loads(line) for line in (tmp_path / "swebench_verified.jsonl").open()]
    assert {r["model"] for r in rows} == {"agent_good", "nodate"}
    assert sum(r["score"] for r in rows if r["model"] == "agent_good") == 3
    assert "skipped, reported 50.0 %" in capsys.readouterr().err


def test_reported_rate_parsing(tmp_path):
    mod = _load()
    _place(tmp_path, "s", [True], 69.6)
    assert mod.reported_rate("s", tmp_path) == pytest.approx(69.6)
    assert mod.label("20260217_mini-v2.0.0_glm-5-high") == "mini-v2.0.0_glm-5-high"


def test_urls_use_the_pinned_ref():
    mod = _load()
    assert len(mod.DEFAULT_REF) == 40
    assert mod.RAW.format(ref=mod.DEFAULT_REF).startswith(
        "https://raw.githubusercontent.com/SWE-bench/experiments/" + mod.DEFAULT_REF
    )
