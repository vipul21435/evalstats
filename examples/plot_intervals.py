"""Plot per-model intervals and paired differences from a results file.

Two panels, rendered headlessly with matplotlib (Agg):

* left: each model's mean with the item-level CLT interval and the
  cluster-robust t interval (clusters from the item-id prefix);
* right: each model's paired difference from the leader with its
  cluster-robust interval, the quantity a ranking should be read from.

    pip install matplotlib
    python examples/plot_intervals.py results.jsonl --out docs/intervals.png
"""

from __future__ import annotations

import argparse
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from evalstats.intervals import mean_interval  # noqa: E402
from evalstats.io import align, read_records, scores_by_model  # noqa: E402

BLUE = "#2a78d6"
ORANGE = "#eb6834"
INK = "#0b0b0b"
MUTED = "#52514e"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("path", help="JSONL or CSV results file with prefixed item ids")
    ap.add_argument("--out", required=True, help="PNG file to write")
    ap.add_argument("--strip", default="", help="prefix to drop from model names in labels")
    args = ap.parse_args(argv)

    models = scores_by_model(read_records(args.path), default_cluster=True)
    rows = []
    for name, m in models.items():
        plain = mean_interval(m.scores)
        clustered = mean_interval(m.scores, clusters=m.clusters)
        rows.append((name, plain, clustered))
    rows.sort(key=lambda r: r[1].estimate)
    leader = rows[-1][0]
    diffs = []
    for name, _, _ in rows:
        if name == leader:
            continue
        _, sa, sb, cl = align(models[name], models[leader])
        diffs.append((name, mean_interval(sa - sb, clusters=cl), mean_interval(sa - sb)))

    def label(n: str) -> str:
        return n[len(args.strip) :] if args.strip and n.startswith(args.strip) else n

    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 0.42 * len(rows) + 1.6), sharey=False)
    for i, (_, plain, cl) in enumerate(rows):
        left.plot([100 * plain.low, 100 * plain.high], [i + 0.14] * 2, color=BLUE, lw=2)
        left.plot([100 * cl.low, 100 * cl.high], [i - 0.14] * 2, color=ORANGE, lw=2)
        left.plot(100 * plain.estimate, i, "o", color=INK, ms=4)
    left.set_yticks(range(len(rows)), [label(r[0]) for r in rows], fontsize=8)
    left.set_xlabel("resolved % (95% interval)", color=MUTED)
    left.set_title("mean per model", fontsize=10, color=INK, loc="left")
    left.plot([], [], color=BLUE, lw=2, label="items independent (CLT)")
    left.plot([], [], color=ORANGE, lw=2, label="clustered by repository (t, G-1)")
    fig.legend(loc="upper center", ncol=2, fontsize=9, frameon=False)

    for i, (_, cl, plain) in enumerate(diffs):
        right.plot([100 * plain.low, 100 * plain.high], [i + 0.14] * 2, color=BLUE, lw=2)
        right.plot([100 * cl.low, 100 * cl.high], [i - 0.14] * 2, color=ORANGE, lw=2)
        right.plot(100 * cl.estimate, i, "o", color=INK, ms=4)
    right.axvline(0, color=MUTED, lw=1, ls="--")
    right.set_yticks(range(len(diffs)), [label(d[0]) for d in diffs], fontsize=8)
    right.set_xlabel("points (95% interval)", color=MUTED)
    right.set_title(
        f"paired difference: model minus {label(leader)}", fontsize=10, color=INK, loc="right"
    )

    for ax in (left, right):
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", color="#e4e3df", lw=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(colors=MUTED)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(args.out, dpi=130)
    print(f"wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
