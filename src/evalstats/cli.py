"""Command line interface: ``evalstats summary`` and ``evalstats compare``."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from evalstats import __version__
from evalstats.intervals import mean_interval
from evalstats.io import align, read_records, scores_by_model
from evalstats.paired import paired_comparison
from evalstats.passk import pass_at_k


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("path", help="JSONL or CSV results file")
    p.add_argument("--level", type=float, default=0.95, help="confidence level (default 0.95)")
    p.add_argument(
        "--method",
        choices=["clt", "percentile", "bca"],
        default="clt",
        help="interval method (default clt)",
    )
    p.add_argument(
        "--cluster-by-prefix",
        action="store_true",
        help="cluster items by the part of the id before the first '__' when no cluster column",
    )
    p.add_argument("--n-boot", type=int, default=2000, help="bootstrap resamples (default 2000)")
    p.add_argument("--seed", type=int, default=0, help="random seed (default 0)")
    p.add_argument("--json", action="store_true", help="print JSON instead of Markdown")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="evalstats", description=__doc__)
    parser.add_argument("--version", action="version", version=f"evalstats {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    s = sub.add_parser("summary", help="mean score with an interval, per model")
    _common(s)

    c = sub.add_parser("compare", help="paired comparison of two models on shared items")
    _common(c)
    c.add_argument("--a", required=True, help="first model name")
    c.add_argument("--b", required=True, help="second model name")
    c.add_argument("--n-perm", type=int, default=10000, help="sign-flip permutations")

    k = sub.add_parser("passk", help="unbiased pass@k from n_samples/n_correct rows, per model")
    k.add_argument("path", help="JSONL or CSV results file with n_samples and n_correct")
    k.add_argument("--k", type=int, default=1, help="k (default 1)")
    k.add_argument("--level", type=float, default=0.95, help="confidence level (default 0.95)")
    k.add_argument("--n-boot", type=int, default=2000, help="bootstrap resamples (default 2000)")
    k.add_argument("--seed", type=int, default=0, help="random seed (default 0)")
    k.add_argument("--json", action="store_true", help="print JSON instead of Markdown")
    return parser


def _pct(x: float) -> str:
    return f"{100 * x:.1f}"


def cmd_summary(args: argparse.Namespace) -> int:
    models = scores_by_model(read_records(args.path), default_cluster=args.cluster_by_prefix)
    rows = []
    for name in sorted(models):
        m = models[name]
        iv = mean_interval(
            m.scores,
            level=args.level,
            method=args.method,
            clusters=m.clusters,
            n_boot=args.n_boot,
            seed=args.seed,
        )
        rows.append({"model": name, **iv.to_dict()})
    if args.json:
        print(json.dumps(rows, indent=2))
        return 0
    lvl = int(round(100 * args.level))
    print(f"| model | n | clusters | mean % | {lvl}% low | {lvl}% high | method |")
    print("|---|---:|---:|---:|---:|---:|---|")
    for r in rows:
        cl = "-" if r["n_clusters"] is None else str(r["n_clusters"])
        print(
            f"| {r['model']} | {r['n']} | {cl} | {_pct(r['estimate'])} "
            f"| {_pct(r['low'])} | {_pct(r['high'])} | {r['method']} |"
        )
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    models = scores_by_model(read_records(args.path), default_cluster=args.cluster_by_prefix)
    for name in (args.a, args.b):
        if name not in models:
            print(f"error: model {name!r} not in {args.path}", file=sys.stderr)
            return 2
    shared, sa, sb, clusters = align(models[args.a], models[args.b])
    res = paired_comparison(
        sa,
        sb,
        level=args.level,
        method=args.method,
        clusters=clusters,
        n_boot=args.n_boot,
        n_perm=args.n_perm,
        seed=args.seed,
    )
    if args.json:
        print(json.dumps({"a": args.a, "b": args.b, **res.to_dict()}, indent=2))
        return 0
    d = res.difference
    lvl = int(round(100 * args.level))
    print(f"shared items: {res.n}")
    print(f"{args.a}: {_pct(res.mean_a)} %")
    print(f"{args.b}: {_pct(res.mean_b)} %")
    print(
        f"difference (a - b): {_pct(d.estimate)} points, "
        f"{lvl}% interval [{_pct(d.low)}, {_pct(d.high)}] ({d.method})"
    )
    print(f"sign-flip permutation p: {res.permutation_p:.4f}")
    if res.mcnemar is not None:
        mc = res.mcnemar
        print(
            f"McNemar exact: a-only {mc.b}, b-only {mc.c}, "
            f"discordant {mc.discordant}, p = {mc.p_value:.4f}"
        )
    return 0


def cmd_passk(args: argparse.Namespace) -> int:
    by_model: dict[str, list[tuple[int, int]]] = {}
    for r in read_records(args.path):
        if r.n_samples is None or r.n_correct is None:
            continue
        by_model.setdefault(r.model, []).append((r.n_samples, r.n_correct))
    if not by_model:
        print("error: no rows with n_samples and n_correct", file=sys.stderr)
        return 2
    rows = []
    for name in sorted(by_model):
        n, c = zip(*by_model[name], strict=True)
        iv = pass_at_k(n, c, args.k, level=args.level, n_boot=args.n_boot, seed=args.seed)
        rows.append({"model": name, "k": args.k, **iv.to_dict()})
    if args.json:
        print(json.dumps(rows, indent=2))
        return 0
    lvl = int(round(100 * args.level))
    print(f"| model | items | pass@{args.k} % | {lvl}% low | {lvl}% high |")
    print("|---|---:|---:|---:|---:|")
    for r in rows:
        print(
            f"| {r['model']} | {r['n']} | {_pct(r['estimate'])} "
            f"| {_pct(r['low'])} | {_pct(r['high'])} |"
        )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "summary":
        return cmd_summary(args)
    if args.command == "compare":
        return cmd_compare(args)
    if args.command == "passk":
        return cmd_passk(args)
    return 2  # pragma: no cover


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
