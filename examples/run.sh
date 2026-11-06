#!/usr/bin/env sh
# Regenerate the synthetic fixture and run both commands on it.
set -e
cd "$(dirname "$0")/.."
python examples/make_fixture.py
evalstats summary examples/results.jsonl --cluster-by-prefix --method bca
evalstats compare examples/results.jsonl --a a --b b --cluster-by-prefix
evalstats compare examples/results.jsonl --a a --b c --cluster-by-prefix
