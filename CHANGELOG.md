# Changelog

## 0.1.0

First version.

- Mean with CLT, percentile-bootstrap and BCa-bootstrap intervals.
- Cluster-robust standard errors for item groups, with a `t(G - 1)` critical
  value for clustered CLT intervals.
- Paired comparison of two models on shared items: paired-difference interval,
  exact McNemar test for binary scores, item-level and cluster-level sign-flip
  permutation tests.
- Unbiased pass@k estimator with a bootstrap interval over items, from
  per-item counts or one row per sample.
- Power and sample-size planning from the observed discordance or the spread of
  per-item differences, with a cluster design effect.
- Leaderboard mode: every model ranked, every pair tested, Holm and
  Benjamini-Hochberg adjusted p-values, and the group not separated from the
  leader.
- `evalstats report`: intervals, corrected pairs and power plans in one
  Markdown or JSON document.
- Commands `summary`, `compare`, `leaderboard`, `power`, `report` and `passk`
  reading JSONL or CSV.
- Examples: a synthetic fixture, a fetch script for public SWE-bench Verified
  submission results, a script reproducing the README results, and an interval
  plot.
