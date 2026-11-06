"""Command line interface.

Commands: ``summary`` (one interval per model), ``compare`` (two models on
shared items), ``leaderboard`` (every model and every pair with corrected
p-values), ``power`` (items needed to detect a difference) and ``passk``.
Every command prints Markdown, or JSON with ``--json``; ``--out`` writes the
report to a file instead of stdout.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from evalstats import __version__
from evalstats.intervals import mean_interval
from evalstats.io import align, passk_counts, read_records, scores_by_model
from evalstats.leaderboard import leaderboard
from evalstats.paired import paired_comparison
from evalstats.passk import pass_at_k
from evalstats.power import plan_from_scores
from evalstats.report import emit, markdown_table, p_value, pct, to_json


def _output(p: argparse.ArgumentParser) -> None:
    p.add_argument("--json", action="store_true", help="print JSON instead of Markdown")
    p.add_argument("--out", help="write the report to this file instead of stdout")


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
    _output(p)


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

    lb = sub.add_parser("leaderboard", help="all models ranked, all pairs tested and corrected")
    _common(lb)
    lb.add_argument("--n-perm", type=int, default=10000, help="sign-flip permutations")
    lb.add_argument(
        "--against-leader",
        action="store_true",
        help="test only the pairs that involve the top model",
    )
    lb.add_argument(
        "--models", nargs="+", help="restrict the leaderboard to these model names"
    )

    pw = sub.add_parser("power", help="items needed to detect a difference between two models")
    pw.add_argument("path", help="JSONL or CSV results file")
    pw.add_argument("--a", required=True, help="first model name")
    pw.add_argument("--b", required=True, help="second model name")
    pw.add_argument(
        "--delta", type=float, default=0.03, help="difference to detect, as a proportion"
    )
    pw.add_argument("--alpha", type=float, default=0.05, help="two-sided size (default 0.05)")
    pw.add_argument("--power", type=float, default=0.8, help="target power (default 0.8)")
    pw.add_argument(
        "--cluster-by-prefix",
        action="store_true",
        help="cluster items by the id prefix before '__' for the design effect",
    )
    _output(pw)

    k = sub.add_parser("passk", help="unbiased pass@k per model")
    k.add_argument(
        "path",
        help="results file with n_samples and n_correct per item, or one row per sample",
    )
    k.add_argument("--k", type=int, default=1, help="k (default 1)")
    k.add_argument("--level", type=float, default=0.95, help="confidence level (default 0.95)")
    k.add_argument("--n-boot", type=int, default=2000, help="bootstrap resamples (default 2000)")
    k.add_argument("--seed", type=int, default=0, help="random seed (default 0)")
    _output(k)
    return parser


def _interval_rows(models, args):
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
    return rows


def _interval_table(rows, level: float) -> str:
    lvl = int(round(100 * level))
    table = [
        [
            r["model"],
            r["n"],
            "-" if r["n_clusters"] is None else r["n_clusters"],
            pct(r["estimate"]),
            pct(r["low"]),
            pct(r["high"]),
            r["method"],
        ]
        for r in rows
    ]
    return markdown_table(
        ["model", "n", "clusters", "mean %", f"{lvl}% low", f"{lvl}% high", "method"], table
    )


def cmd_summary(args: argparse.Namespace) -> int:
    models = scores_by_model(read_records(args.path), default_cluster=args.cluster_by_prefix)
    rows = _interval_rows(models, args)
    if args.json:
        emit(to_json(rows), args.out)
    else:
        emit(_interval_table(rows, args.level), args.out)
    return 0


def _two_models(args):
    models = scores_by_model(read_records(args.path), default_cluster=args.cluster_by_prefix)
    for name in (args.a, args.b):
        if name not in models:
            print(f"error: model {name!r} not in {args.path}", file=sys.stderr)
            return None
    return align(models[args.a], models[args.b])


def cmd_compare(args: argparse.Namespace) -> int:
    aligned = _two_models(args)
    if aligned is None:
        return 2
    _, sa, sb, clusters = aligned
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
        emit(to_json({"a": args.a, "b": args.b, **res.to_dict()}), args.out)
        return 0
    d = res.difference
    lvl = int(round(100 * args.level))
    lines = [
        f"shared items: {res.n}",
        f"{args.a}: {pct(res.mean_a)} %",
        f"{args.b}: {pct(res.mean_b)} %",
        f"difference (a - b): {pct(d.estimate)} points, "
        f"{lvl}% interval [{pct(d.low)}, {pct(d.high)}] ({d.method})",
        f"sign-flip permutation p: {p_value(res.permutation_p)}",
    ]
    if res.mcnemar is not None:
        mc = res.mcnemar
        lines.append(
            f"McNemar exact: a-only {mc.b}, b-only {mc.c}, "
            f"discordant {mc.discordant}, p = {p_value(mc.p_value)}"
        )
    emit("\n".join(lines), args.out)
    return 0


def cmd_leaderboard(args: argparse.Namespace) -> int:
    models = scores_by_model(read_records(args.path), default_cluster=args.cluster_by_prefix)
    if args.models:
        missing = [m for m in args.models if m not in models]
        if missing:
            print(f"error: models not in {args.path}: {', '.join(missing)}", file=sys.stderr)
            return 2
        models = {m: models[m] for m in args.models}
    lb = leaderboard(
        models,
        level=args.level,
        method=args.method,
        n_boot=args.n_boot,
        n_perm=args.n_perm,
        seed=args.seed,
        against_leader=args.against_leader,
    )
    if args.json:
        emit(to_json(lb.to_dict()), args.out)
        return 0
    lvl = int(round(100 * args.level))
    rank_rows = [
        [
            i + 1,
            name,
            iv.n,
            "-" if iv.n_clusters is None else iv.n_clusters,
            pct(iv.estimate),
            pct(iv.low),
            pct(iv.high),
        ]
        for i, (name, iv) in enumerate(lb.models)
    ]
    ranks = markdown_table(
        ["rank", "model", "n", "clusters", "mean %", f"{lvl}% low", f"{lvl}% high"], rank_rows
    )
    pair_rows = [
        [
            r.a,
            r.b,
            r.n,
            pct(r.difference),
            r.test,
            p_value(r.p_value),
            p_value(r.p_holm),
            p_value(r.p_bh),
        ]
        for r in lb.pairs
    ]
    pairs = markdown_table(
        ["a", "b", "n", "a - b (points)", "test", "p", "p Holm", "p BH"], pair_rows
    )
    top = ", ".join(lb.top_group())
    text = (
        f"{ranks}\n\n{pairs}\n\n"
        f"not separated from the leader at Holm-adjusted alpha {lb.alpha:g}: {top}"
    )
    emit(text, args.out)
    return 0


def cmd_power(args: argparse.Namespace) -> int:
    aligned = _two_models(args)
    if aligned is None:
        return 2
    _, sa, sb, clusters = aligned
    plan = plan_from_scores(
        sa, sb, args.delta, clusters=clusters, alpha=args.alpha, power=args.power
    )
    if args.json:
        emit(to_json({"a": args.a, "b": args.b, **plan.to_dict()}), args.out)
        return 0
    lines = [f"shared items: {plan.n_observed}"]
    if plan.discordance is not None:
        lines.append(f"observed discordance: {pct(plan.discordance)} % of items")
    lines += [
        f"sd of per-item differences: {plan.sd_difference:.4f}",
        f"design effect from clusters: {plan.design_effect:.2f}",
        f"items to detect {pct(plan.delta)} points at alpha {plan.alpha:g}, "
        f"power {plan.power:g}: {plan.n_items} (independent items), "
        f"{plan.n_items_clustered} (with the design effect)",
    ]
    if plan.power_at_observed is not None and plan.detectable_at_observed is not None:
        lines += [
            f"power for {pct(plan.delta)} points on the observed {plan.n_observed} items: "
            f"{plan.power_at_observed:.2f}",
            f"smallest difference detectable on {plan.n_observed} items: "
            f"{pct(plan.detectable_at_observed)} points",
        ]
    emit("\n".join(lines), args.out)
    return 0


def cmd_passk(args: argparse.Namespace) -> int:
    by_model = passk_counts(read_records(args.path))
    if not by_model:
        print("error: no rows with n_samples and n_correct, or per-sample scores", file=sys.stderr)
        return 2
    rows = []
    for name in sorted(by_model):
        n, c = zip(*by_model[name], strict=True)
        iv = pass_at_k(n, c, args.k, level=args.level, n_boot=args.n_boot, seed=args.seed)
        rows.append({"model": name, "k": args.k, **iv.to_dict()})
    if args.json:
        emit(to_json(rows), args.out)
        return 0
    lvl = int(round(100 * args.level))
    table = [[r["model"], r["n"], pct(r["estimate"]), pct(r["low"]), pct(r["high"])] for r in rows]
    headers = ["model", "items", f"pass@{args.k} %", f"{lvl}% low", f"{lvl}% high"]
    emit(markdown_table(headers, table), args.out)
    return 0


COMMANDS = {
    "summary": cmd_summary,
    "compare": cmd_compare,
    "leaderboard": cmd_leaderboard,
    "power": cmd_power,
    "passk": cmd_passk,
}


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return COMMANDS[args.command](args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
