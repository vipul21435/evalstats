#!/usr/bin/env sh
# Regenerate the synthetic fixture and run the quickstart commands on it.
set -e
cd "$(dirname "$0")/.."
python examples/make_fixture.py
evalstats summary examples/results.jsonl --cluster-by-prefix --method bca
evalstats compare examples/results.jsonl --a a --b b
evalstats compare examples/results.jsonl --a a --b c
evalstats compare examples/results.jsonl --a a --b b --cluster-by-prefix
evalstats power examples/results.jsonl --a a --b b --delta 0.03
evalstats power --discordance 0.12 --delta 0.03
evalstats leaderboard examples/results.jsonl
