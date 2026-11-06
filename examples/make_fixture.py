"""Write a small synthetic results file, examples/results.jsonl.

Three models on 120 items spread over 8 clusters. Model ``b`` is model ``a``
with a few extra solves; model ``c`` is independent of both. Scores are 0/1.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def main(seed: int = 7, n_items: int = 120, n_clusters: int = 8) -> Path:
    rng = np.random.default_rng(seed)
    cluster_rate = rng.uniform(0.2, 0.7, size=n_clusters)
    cluster_of = rng.integers(0, n_clusters, size=n_items)
    a = (rng.random(n_items) < cluster_rate[cluster_of]).astype(int)
    flip = rng.random(n_items) < 0.08
    b = np.where(flip & (a == 0), 1, a)
    c = (rng.random(n_items) < cluster_rate[cluster_of]).astype(int)
    out = HERE / "results.jsonl"
    with out.open("w", encoding="utf-8") as fh:
        for i in range(n_items):
            item = f"repo{cluster_of[i]:02d}__{i + 1:04d}"
            for model, s in (("a", a[i]), ("b", b[i]), ("c", c[i])):
                fh.write(json.dumps({"item": item, "model": model, "score": int(s)}) + "\n")
    return out


if __name__ == "__main__":
    print(main())
