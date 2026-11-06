"""Download per-instance results of public SWE-bench Verified submissions.

The public experiments repository (https://github.com/SWE-bench/experiments)
keeps one ``per_instance_details.json`` per submission under
``evaluation/verified/<submission>/`` with a ``resolved`` flag per instance.
This script fetches those small files for a list of submissions and writes a
single JSONL results file with one row per (instance, submission):
``item``, ``model``, ``score``. Instance ids are ``<owner>__<repo>-<number>``,
so ``--cluster-by-prefix`` in evalstats clusters them by repository.

The experiments repository publishes no licence file, so nothing from it is
redistributed here: run this script to reproduce the results in the README.

    python examples/fetch_swebench.py --out ~/data/evalstats
"""

from __future__ import annotations

import argparse
import json
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
    "20260226_mini-v2.0.0_gemini-3-pro-high",
]


def label(submission: str) -> str:
    """Drop the date prefix of a submission folder name."""
    return submission.split("_", 1)[1] if "_" in submission else submission


def fetch(submission: str, cache: Path) -> dict:
    path = cache / f"{submission}.json"
    if not path.exists():
        url = f"{RAW}/{submission}/per_instance_details.json"
        with urllib.request.urlopen(url, timeout=60) as resp:
            path.write_bytes(resp.read())
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True, help="directory for the cache and the JSONL file")
    ap.add_argument("--submissions", nargs="+", default=DEFAULT_SUBMISSIONS)
    args = ap.parse_args(argv)
    out = Path(args.out).expanduser()
    cache = out / "submissions"
    cache.mkdir(parents=True, exist_ok=True)
    target = out / "swebench_verified.jsonl"
    n_rows = 0
    with target.open("w", encoding="utf-8") as fh:
        for sub in args.submissions:
            details = fetch(sub, cache)
            for item in sorted(details):
                score = int(bool(details[item]["resolved"]))
                row = {"item": item, "model": label(sub), "score": score}
                fh.write(json.dumps(row) + "\n")
                n_rows += 1
            print(f"{sub}: {len(details)} instances", file=sys.stderr)
    print(f"wrote {n_rows} rows to {target}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
