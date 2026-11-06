"""Error bars and paired comparisons for model evaluations."""

__version__ = "0.1.0"

from evalstats.intervals import Interval, Method, mean_interval, standard_error
from evalstats.io import ModelScores, Record, align, read_records, scores_by_model
from evalstats.leaderboard import Leaderboard, benjamini_hochberg, holm, leaderboard
from evalstats.paired import (
    McNemar,
    PairedResult,
    cluster_sign_flip_test,
    mcnemar_exact,
    paired_comparison,
    sign_flip_test,
)
from evalstats.passk import pass_at_k, pass_at_k_item
from evalstats.power import PowerPlan, items_for_discordance, items_for_sd, plan_from_scores

__all__ = [
    "Interval",
    "Leaderboard",
    "McNemar",
    "Method",
    "ModelScores",
    "PairedResult",
    "PowerPlan",
    "Record",
    "__version__",
    "align",
    "benjamini_hochberg",
    "cluster_sign_flip_test",
    "holm",
    "items_for_discordance",
    "items_for_sd",
    "leaderboard",
    "mcnemar_exact",
    "mean_interval",
    "paired_comparison",
    "pass_at_k",
    "pass_at_k_item",
    "plan_from_scores",
    "read_records",
    "scores_by_model",
    "sign_flip_test",
    "standard_error",
]
