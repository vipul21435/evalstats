# Changelog

## Unreleased

- Bad input (missing file, malformed JSONL or CSV row, missing column,
  out-of-range flag) prints a one-line `error:` message and exits with code 2;
  row errors name the file and line.
- BCa bias correction counts bootstrap means equal to the estimate at half
  weight; checked against `scipy.stats.bootstrap(method="BCa")`.
- `pass_at_k_item` rejects counts outside `0 <= c <= n` and `k < 1`.
- `compare` names the two models in its output; summary tables show `-` for
  bounds that need at least two items.
- The public functions are re-exported from `evalstats`, array arguments are
  typed with `numpy.typing.ArrayLike`, and the package ships `py.typed`.
- `examples/fetch_swebench.py` pins the experiments repository commit and takes
  `--ref`; `examples/coverage.py` prints the coverage simulation figures.

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
