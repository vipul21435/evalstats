#!/usr/bin/env sh
# Reproduce the README's results on public SWE-bench Verified submissions.
# Usage: sh examples/swebench.sh DATA_DIR
# Downloads about 0.7 MB of per-instance result files into DATA_DIR.
set -e
DATA="${1:?usage: sh examples/swebench.sh DATA_DIR}"
cd "$(dirname "$0")/.."
python examples/fetch_swebench.py --out "$DATA"
F="$DATA/swebench_verified.jsonl"
A=mini-v2.0.0_gemini-3-flash-high
B=mini-v2.0.0_glm-5-high
OPUS=mini-v2.0.0_claude-4-5-opus-high
evalstats summary "$F"
evalstats summary "$F" --cluster-by-prefix
evalstats compare "$F" --a "$A" --b "$B"
evalstats compare "$F" --a "$A" --b "$B" --cluster-by-prefix
evalstats power "$F" --a "$A" --b "$B" --delta 0.03 --cluster-by-prefix
evalstats compare "$F" --a "$OPUS" --b "$B" --cluster-by-prefix
evalstats power "$F" --a "$OPUS" --b "$B" --delta 0.03 --cluster-by-prefix
evalstats leaderboard "$F" --cluster-by-prefix --against-leader
evalstats leaderboard "$F" --cluster-by-prefix
evalstats leaderboard "$F" --cluster-by-prefix --test cluster
evalstats summary "$F" --cluster-by-prefix --method bca
python -c "import json,sys,collections;c=collections.Counter(json.loads(l)['item'].split('__')[0] for l in open(sys.argv[1]) if '4-5-opus' in l);print(c.most_common(3))" "$F"
evalstats report "$F" --cluster-by-prefix --out "$DATA/report.md"
python examples/plot_intervals.py "$F" --out docs/swebench_intervals.png --strip mini-v2.0.0_ \
  --rename mini-v1.17.2_gpt-5.2-2025-12-11-high=v1.17.2_gpt-5.2
