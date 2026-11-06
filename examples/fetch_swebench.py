"""Download per-instance results of public SWE-bench Verified submissions.

The public experiments repository (https://github.com/SWE-bench/experiments)
keeps one ``per_instance_details.json`` per submission under
``evaluation/verified/<submission>/`` with a ``resolved`` flag per instance.
This script fetches those small files for a list of submissions and writes a
single JSONL results file with one row per (instance, submission):
``item``, ``model``, ``score``. Instance ids are ``<owner>__<repo>-<number>``,
so ``--cluster-by-prefix`` in evalstats clusters them by repository.

Each submission also has a ``metadata.yaml`` with the resolved rate reported
on the leaderboard. The script compares it with the rate computed from the
per-instance file and skips a submission when the two differ by more than
``--tolerance`` points. When this script was written, the per-instance file of
``20260226_mini-v2.0.0_gemini-3-pro-high`` marked every instance unresolved
while its metadata reported 69.6 %, so that submission is left out of the
default list and would be skipped if named.

The experiments repository publishes no licence file, so nothing from it is
redistributed here: run this script to reproduce the results in the README.

    python examples/fetch_swebench.py --out ~/data/evalstats
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from pathlib import Path

RAW = "https://raw.githubusercontent.com/SWE-bench/experiments/main/evaluation/verified"

DEFAULT_SUBMISSIONS = [
    "20251211_mini-v1.17.2_gpt-5.2-2025-12-11-high",
    "20260217_mini-v2.0.0_claude-4-5-haiku-high",
    "20260217_mini-v2.0.0_claude-4-5-opus-high",
    "20260217_mini-v2.0.0_claude-4-5-sonnet-high",
    "20260217_mini-v2.0.0_claude-4-6-opus",
    "20260217_mini-v2.0.0_deepseek-3-2-high",
    "20260217_mini-v2.0.0_gemini-3-flash-high",
    "20260217_mini-v2.0.0_glm-5-high",
    "20260217_mini-v2.0.0_gpt-5-2-high",
    "20260217_mini-v2.0.0_gpt-5-mini",
    "20260217_mini-v2.0.0_kimi-k2-5-high",
    "20260217_mini-v2.0.0_minimax-2-5-high",
]


def label(submission: str) -> str:
    """Drop the date prefix of a submission folder name."""
    return submission.split("_", 1)[1] if "_" in submission else submission


def _cached(url: str, path: Path) -> str:
    if not path.exists():
        with urllib.request.urlopen(url, timeout=60) as resp:
            path.write_bytes(resp.read())
    return path.read_text(encoding="utf-8")


def fetch(submission: str, cache: Path) -> dict:
    """The per-instance details of one submission, cached on disk."""
    url = f"{RAW}/{submission}/per_instance_details.json"
    return json.loads(_cached(url, cache / f"{submission}.json"))


def reported_rate(submission: str, cache: Path) -> float | None:
    """The ``resolved`` percentage in the submission's metadata, if present."""
    url = f"{RAW}/{submission}/metadata.yaml"
    text = _cached(url, cache / f"{submission}.metadata.yaml")
    m = re.search(r"^\s*resolved:\s*([0-9.]+)\s*$", text, flags=re.MULTILINE)
    return float(m.group(1)) if m else None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True, help="directory for the cache and the JSONL file")
    ap.add_argument("--submissions", nargs="+", default=DEFAULT_SUBMISSIONS)
    ap.add_argument(
        "--tolerance",
        type=float,
        default=0.2,
        help="largest gap in points between reported and computed rate (default 0.2)",
    )
    args = ap.parse_args(argv)
    out = Path(args.out).expanduser()
    cache = out / "submissions"
    cache.mkdir(parents=True, exist_ok=True)
    target = out / "swebench_verified.jsonl"
    n_rows = 0
    with target.open("w", encoding="utf-8") as fh:
        for sub in args.submissions:
            details = fetch(sub, cache)
            computed = 100 * sum(bool(v["resolved"]) for v in details.values()) / len(details)
            reported = reported_rate(sub, cache)
            if reported is not None and abs(reported - computed) > args.tolerance:
                print(
                    f"{sub}: skipped, reported {reported:.1f} % but per-instance file "
                    f"gives {computed:.1f} %",
                    file=sys.stderr,
                )
                continue
            for item in sorted(details):
                score = int(bool(details[item]["resolved"]))
                row = {"item": item, "model": label(sub), "score": score}
                fh.write(json.dumps(row) + "\n")
                n_rows += 1
            print(f"{sub}: {len(details)} instances, {computed:.1f} % resolved", file=sys.stderr)
    print(f"wrote {n_rows} rows to {target}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
