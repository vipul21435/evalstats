# Changelog

## 0.1.0 - 2026-11-06

### Added

- Mean of per-item scores with CLT, percentile-bootstrap and BCa-bootstrap
  intervals; BCa is checked against `scipy.stats.bootstrap`.
- Cluster-robust standard errors and a cluster bootstrap for item groups, with
  a `t(G - 1)` critical value for clustered CLT intervals.
- Paired comparison of two models on shared items: paired-difference interval,
  exact McNemar test for binary scores, and item-level and cluster-level
  sign-flip permutation tests.
- Unbiased pass@k estimator with a bootstrap interval over items, from
  per-item counts or one row per sample.
- Power and sample-size planning from the observed discordance or the spread of
  per-item differences, with a cluster design effect.
- Leaderboard mode: every model ranked, every pair tested, Holm and
  Benjamini-Hochberg adjusted p-values, and the group not separated from the
  leader.
- Commands `summary`, `compare`, `leaderboard`, `power`, `report` and `passk`
  reading JSONL or CSV; `report` writes intervals, corrected pairs and power
  plans as one Markdown or JSON document.
- Input checks: bad input prints a one-line `error:` message naming the file
  and line, and exits with code 2; NaN and infinite scores, inconsistent
  clusters and malformed rows are rejected.
- Bootstrap resamples and sign flips are drawn in chunks, so memory does not
  grow with the number of resamples times the item count.
- Python API re-exported from `evalstats`, typed with `numpy.typing.ArrayLike`,
  shipped with `py.typed`; a `plot` extra installs matplotlib.
- Examples: a synthetic fixture, `examples/fetch_swebench.py` for public
  SWE-bench Verified submission results at a pinned commit,
  `examples/swebench.sh` reproducing the README results,
  `examples/coverage.py` for the cluster-interval coverage simulation, and an
  interval plot.
- README with a worked quickstart and results on real public data.
- Tests for the sign-flip p-values, the cluster bootstrap and BCa (against slow
  loops and scipy), the design effect and the quickstart outputs; ruff and mypy
  run in CI.

### Known limitations

See the Limitations section of the README, including small-cluster behaviour,
no wild cluster bootstrap, and results from one benchmark and one scaffold.
