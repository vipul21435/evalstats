"""Command line interface.

Commands: ``summary`` (one interval per model), ``compare`` (two models on
shared items), ``leaderboard`` (every model and every pair with corrected
p-values), ``report`` (all of that plus power plans in one document),
``power`` (items needed to detect a difference) and ``passk``.
Every command prints Markdown, or JSON with ``--json``; ``--out`` writes the
report to a file instead of stdout.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from evalstats import __version__
from evalstats.intervals import mean_interval
from evalstats.io import align, passk_counts, read_records, scores_by_model
from evalstats.leaderboard import leaderboard
from evalstats.paired import paired_comparison
from evalstats.passk import pass_at_k
from evalstats.power import items_for_discordance, plan_from_scores
from evalstats.report import (
    emit,
    full_report,
    leaderboard_tables,
    markdown_table,
    p_value,
    pct,
    render_markdown,
    report_json,
    to_json,
)


def _unit_interval(text: str) -> float:
    """argparse type: a number strictly between 0 and 1."""
    try:
        v = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a number: {text!r}") from None
    if not 0 < v < 1:
        raise argparse.ArgumentTypeError(f"must be between 0 and 1, got {text}")
    return v


def _positive_float(text: str) -> float:
    try:
        v = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a number: {text!r}") from None
    if not v > 0:
        raise argparse.ArgumentTypeError(f"must be positive, got {text}")
    return v


def _positive_int(text: str) -> int:
    try:
        v = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not an integer: {text!r}") from None
    if v < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, got {text}")
    return v


def _output(p: argparse.ArgumentParser) -> None:
    p.add_argument("--json", action="store_true", help="print JSON instead of Markdown")
    p.add_argument("--out", help="write the report to this file instead of stdout")


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("path", help="JSONL or CSV results file")
    p.add_argument(
        "--level", type=_unit_interval, default=0.95, help="confidence level (default 0.95)"
    )
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
    p.add_argument(
        "--n-boot", type=_positive_int, default=2000, help="bootstrap resamples (default 2000)"
    )
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
    c.add_argument("--n-perm", type=_positive_int, default=10000, help="sign-flip permutations")

    lb = sub.add_parser("leaderboard", help="all models ranked, all pairs tested and corrected")
    _common(lb)
    lb.add_argument("--n-perm", type=_positive_int, default=10000, help="sign-flip permutations")
    lb.add_argument(
        "--against-leader",
        action="store_true",
        help="test only the pairs that involve the top model",
    )
    lb.add_argument("--models", nargs="+", help="restrict the leaderboard to these model names")
    lb.add_argument(
        "--test",
        choices=["auto", "mcnemar", "sign-flip", "cluster"],
        default="auto",
        help="per-pair test: auto (McNemar for 0/1 scores, else sign-flip), "
        "or cluster (sign flips of whole clusters; needs clusters)",
    )

    rp = sub.add_parser("report", help="intervals, corrected pairs and power plans in one report")
    _common(rp)
    rp.add_argument("--n-perm", type=_positive_int, default=10000, help="sign-flip permutations")
    rp.add_argument(
        "--test",
        choices=["auto", "mcnemar", "sign-flip", "cluster"],
        default="auto",
        help="per-pair test, as in leaderboard",
    )
    rp.add_argument(
        "--delta", type=_positive_float, default=0.03, help="difference to plan for (default 0.03)"
    )
    rp.add_argument("--power", type=_unit_interval, default=0.8, help="target power (default 0.8)")

    pw = sub.add_parser("power", help="items needed to detect a difference between two models")
    pw.add_argument("path", nargs="?", help="JSONL or CSV results file (omit with --discordance)")
    pw.add_argument("--a", help="first model name (with a results file)")
    pw.add_argument("--b", help="second model name (with a results file)")
    pw.add_argument(
        "--discordance",
        type=_unit_interval,
        help="plan without data: assumed share of items the two models disagree on",
    )
    pw.add_argument(
        "--delta", type=_positive_float, default=0.03, help="difference to detect, as a proportion"
    )
    pw.add_argument(
        "--alpha", type=_unit_interval, default=0.05, help="two-sided size (default 0.05)"
    )
    pw.add_argument("--power", type=_unit_interval, default=0.8, help="target power (default 0.8)")
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
    k.add_argument("--k", type=_positive_int, default=1, help="k (default 1)")
    k.add_argument(
        "--level", type=_unit_interval, default=0.95, help="confidence level (default 0.95)"
    )
    k.add_argument(
        "--n-boot", type=_positive_int, default=2000, help="bootstrap resamples (default 2000)"
    )
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
    if res.cluster_permutation_p is not None:
        lines.append(
            f"cluster sign-flip p ({res.n_clusters} clusters): {p_value(res.cluster_permutation_p)}"
        )
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
        test=args.test,
    )
    if args.json:
        emit(to_json(lb.to_dict()), args.out)
        return 0
    text = leaderboard_tables(lb)
    emit(text, args.out)
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    models = scores_by_model(read_records(args.path), default_cluster=args.cluster_by_prefix)
    rep = full_report(
        models,
        source=Path(args.path).name,
        delta=args.delta,
        power=args.power,
        level=args.level,
        method=args.method,
        n_boot=args.n_boot,
        n_perm=args.n_perm,
        seed=args.seed,
        test=args.test,
    )
    emit(report_json(rep) if args.json else render_markdown(rep), args.out)
    return 0


def _power_without_data(args: argparse.Namespace) -> int:
    n = items_for_discordance(args.delta, args.discordance, alpha=args.alpha, power=args.power)
    payload = {
        "delta": args.delta,
        "discordance": args.discordance,
        "alpha": args.alpha,
        "power": args.power,
        "n_items": n,
    }
    if args.json:
        emit(to_json(payload), args.out)
    else:
        emit(
            f"items to detect {pct(args.delta)} points at alpha {args.alpha:g}, "
            f"power {args.power:g}, discordance {pct(args.discordance)} %: {n}",
            args.out,
        )
    return 0


def cmd_power(args: argparse.Namespace) -> int:
    if args.path is None:
        if args.discordance is None:
            print("error: give a results file with --a and --b, or --discordance", file=sys.stderr)
            return 2
        return _power_without_data(args)
    if args.a is None or args.b is None:
        print("error: --a and --b are needed with a results file", file=sys.stderr)
        return 2
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
    "report": cmd_report,
    "power": cmd_power,
    "passk": cmd_passk,
}


def main(argv: Sequence[str] | None = None) -> int:
    """Run one command; bad input prints ``error: ...`` and returns 2."""
    args = build_parser().parse_args(argv)
    try:
        return COMMANDS[args.command](args)
    except (OSError, ValueError, KeyError) as exc:
        if isinstance(exc, OSError) and exc.filename is not None:
            msg = f"{exc.filename}: {exc.strerror}"
        elif isinstance(exc, KeyError) and exc.args:
            msg = exc.args[0]
        else:
            msg = exc
        print(f"error: {msg}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
