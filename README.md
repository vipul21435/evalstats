# evalstats

Error bars and paired comparisons for model evaluations.

Status: first version. The core estimators and the CLI work on per-item result
files. Power planning, the leaderboard mode with multiple-comparison
corrections and the results on public data are not in this version yet (see
Limitations).

## Why this is hard

Two models scored on the same benchmark almost never get the same number, and
the gap is often smaller than the noise in either number. A benchmark is a
sample of items; the items are usually not independent (several come from the
same repository or task family); and when two models are evaluated on the same
items, the right question is not whether their two intervals overlap but
whether the per-item differences are consistently in one direction. A
3-point gap can be strong evidence of a real difference on one benchmark and
no evidence at all on another, depending on how many items the two models
disagree on. evalstats computes the quantities that separate the two cases.

## Install

```
pip install git+https://github.com/vipul21435/evalstats
```

## Quickstart

The input is a JSONL or CSV file with one row per item and model: `item`,
`model`, `score` (0/1 or real) and an optional `cluster` column. With
`--cluster-by-prefix` the cluster is the part of the item id before the first
`__`, the convention several benchmarks use to prefix items with their source
repository.

`examples/make_fixture.py` writes a synthetic file with three models on 120
items in 8 clusters. Model `b` is model `a` with six extra solves; model `c` is
drawn independently.

```
$ evalstats summary examples/results.jsonl --cluster-by-prefix --method bca
| model | n | clusters | mean % | 95% low | 95% high | method |
|---|---:|---:|---:|---:|---:|---|
| a | 120 | 8 | 43.3 | 34.3 | 53.0 | bca |
| b | 120 | 8 | 48.3 | 39.0 | 60.6 | bca |
| c | 120 | 8 | 46.7 | 31.3 | 60.4 | bca |
```

Each model's interval is wide and all three overlap. The paired comparison
tells a different story for the two pairs:

```
$ evalstats compare examples/results.jsonl --a a --b b --cluster-by-prefix
shared items: 120
a: 43.3 %
b: 48.3 %
difference (a - b): -5.0 points, 95% interval [-9.4, -0.6] (clt)
sign-flip permutation p: 0.0332
McNemar exact: a-only 0, b-only 6, discordant 6, p = 0.0312

$ evalstats compare examples/results.jsonl --a a --b c --cluster-by-prefix
shared items: 120
a: 43.3 %
c: 46.7 %
difference (a - b): -3.3 points, 95% interval [-14.8, 8.1] (clt)
sign-flip permutation p: 0.6860
McNemar exact: a-only 25, b-only 29, discordant 54, p = 0.6835
```

`b` beats `a` by 5 points on 6 discordant items, all in the same direction:
the paired interval excludes zero. `c` beats `a` by 3.3 points on 54
discordant items that split 25 to 29: no evidence of a difference. The
marginal intervals in the summary table cannot tell these cases apart; the
paired analysis can.

Add `--json` to any command for machine-readable output.

## How it works

- **Mean with CLT interval**: `mean ± z · se`, with `se = sd / sqrt(n)`.
- **Cluster-robust standard error**: residuals are summed within each cluster
  and the variance of those sums is used, with the `G / (G - 1)` factor for
  `G` clusters. With one item per cluster this equals the plain standard
  error.
- **Percentile bootstrap**: resample items (or whole clusters when a cluster
  key is given) with replacement; take the `α/2` and `1 - α/2` quantiles of
  the resampled means.
- **BCa bootstrap**: the percentile bootstrap with a bias correction `z0`
  from the share of resampled means below the estimate and an acceleration
  `a` from the jackknife skewness of the mean (Efron's bias-corrected and
  accelerated interval). When the bootstrap distribution is degenerate the
  bounds fall back to the percentile bounds.
- **Paired difference**: a one-sample interval on the per-item differences
  `a - b`, by any of the methods above.
- **Exact McNemar**: for 0/1 scores, count the items only `a` solved (`b`) and
  only `b` solved (`c`); under the null the discordant items split as
  `Binomial(b + c, 1/2)`, and the two-sided p-value is twice the lower tail
  at `min(b, c)`, capped at 1.
- **Sign-flip permutation**: flip the sign of each per-item difference at
  random, recompute the mean difference, and report the share of flips at
  least as extreme as the observed one, as `(count + 1) / (n_perm + 1)`.
- **pass@k**: per item `1 - C(n - c, k) / C(n, k)` for `c` correct out of `n`
  samples, averaged over items, with a percentile bootstrap over items.

## Results on real public data

Not in this version. The planned evaluation applies the tool to per-instance
resolved lists of public submissions to a software-engineering benchmark,
clustered by source repository.

## Limitations

- Power and sample-size planning, the leaderboard mode and the Holm and
  Benjamini-Hochberg corrections are not implemented yet.
- The pass@k estimator is available from Python (`evalstats.passk`) but not
  from the CLI yet.
- The CLT interval is a normal approximation and is poor for very small `n`
  or means near 0 or 1; use `--method bca` there.
- Cluster-robust errors need enough clusters (tens, not a handful) to be
  reliable.
- No real-data results have been produced with this version.

## Related tools

- https://github.com/vipul21435/verifybench
- https://github.com/vipul21435/graderguard
- https://github.com/vipul21435/judgebias
- https://github.com/vipul21435/testlogs

## Licence

MIT. See `LICENSE`.

Built by Vipul Raj Jha · https://vipul21435.github.io
