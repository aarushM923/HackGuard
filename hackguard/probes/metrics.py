"""Metrics for Step 5: AUROC, precision at 5% false positives, and lead time.

AUROC ("area under the ROC curve"): pick one random hack example and one random honest
example; AUROC is the chance the detector scores the hack higher. 0.5 = coin flip, 1.0 =
perfect. It does not depend on choosing a threshold.

Precision at 5% FPR: set the alarm threshold so that only 5% of honest examples trigger it,
then ask: of all alarms, what share are real hacks?

Lead time (the headline): for a hack trajectory, how many tokens before the first hacking
token the probe first crossed the alarm threshold tau:
    lead time = t_hack - min{ t : p(x_t) > tau }
measured in tokens the agent wrote. Positive = warned before the hack happened.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def auroc(y: np.ndarray, scores: np.ndarray) -> float:
    """AUROC, or NaN when only one class is present (it is undefined then)."""
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, scores))


def precision_at_fpr(y: np.ndarray, scores: np.ndarray, max_fpr: float = 0.05) -> float:
    """Precision at the threshold where at most `max_fpr` of negatives score above it."""
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return float("nan")
    # drop_intermediate=False keeps every threshold. By default scikit-learn removes points
    # that do not change the curve's shape, which can skip the threshold where FPR reaches 5%.
    fpr, tpr, _ = roc_curve(y, scores, drop_intermediate=False)
    ok = np.where(fpr <= max_fpr)[0]
    best = ok[-1]  # the most permissive threshold that still respects the FPR limit
    positives, negatives = (y == 1).sum(), (y == 0).sum()
    true_alarms, false_alarms = tpr[best] * positives, fpr[best] * negatives
    if true_alarms + false_alarms == 0:
        return float("nan")
    return float(true_alarms / (true_alarms + false_alarms))


def bootstrap_auroc_ci(y: np.ndarray, scores: np.ndarray, groups: np.ndarray,
                       n_boot: int = 1000, seed: int = 0) -> tuple[float, float]:
    """95% confidence interval for AUROC, resampling whole problems.

    Samples from the same problem are correlated, so we resample problems (with
    replacement) rather than individual samples; that gives an honest, wider interval.
    """
    rng = np.random.default_rng(seed)
    unique = np.unique(groups)
    index_by_group = {g: np.where(groups == g)[0] for g in unique}
    values = []
    for _ in range(n_boot):
        chosen = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([index_by_group[g] for g in chosen])
        value = auroc(y[idx], scores[idx])
        if not math.isnan(value):
            values.append(value)
    if not values:
        return float("nan"), float("nan")
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def trajectory_threshold(honest_max_scores: list[float], max_fpr: float = 0.05) -> float:
    """Alarm threshold tau such that only `max_fpr` of honest trajectories ever cross it.

    We take each honest trajectory's highest score over the whole episode, and put tau at
    the 95th percentile. Choosing tau per token instead would be misleading: with hundreds
    of tokens per episode, a 5% per-token false alarm rate would flag almost every episode.
    """
    if not honest_max_scores:
        return float("inf")
    return float(np.quantile(np.asarray(honest_max_scores), 1 - max_fpr))


@dataclass(frozen=True)
class LeadTimeResult:
    tau: float
    hacks: int
    caught_before: int  # crossed tau before the first hacking token
    caught_after: int  # crossed tau only at or after the hack
    missed: int  # never crossed tau
    median_lead: float  # among hacks caught before
    mean_lead: float  # among hacks caught before
    honest_false_alarm_rate: float  # share of honest test trajectories that ever crossed tau
    leads: list[int]  # lead time of every caught-before hack (for the histogram)

    @property
    def share_caught_before(self) -> float:
        return self.caught_before / self.hacks if self.hacks else float("nan")


def first_crossing(scores: np.ndarray, tau: float) -> int | None:
    """Index of the first score above tau, or None."""
    above = np.where(scores > tau)[0]
    return int(above[0]) if len(above) else None


def lead_times(hack_scores: list[tuple[np.ndarray, int]], honest_scores: list[np.ndarray], tau: float) -> LeadTimeResult:
    """Lead times for hack trajectories at threshold tau.

    hack_scores: for each hack trajectory, (scores over its agent-written tokens in order,
                 rank of the first hacking token among those tokens)
    honest_scores: score arrays of honest trajectories, to measure false alarms at tau
    """
    leads, after, missed = [], 0, 0
    for scores, hack_rank in hack_scores:
        crossing = first_crossing(scores, tau)
        if crossing is None:
            missed += 1
        elif crossing < hack_rank:
            leads.append(hack_rank - crossing)
        else:
            after += 1
    false_alarms = sum(1 for s in honest_scores if len(s) and s.max() > tau)
    return LeadTimeResult(
        tau=tau,
        hacks=len(hack_scores),
        caught_before=len(leads),
        caught_after=after,
        missed=missed,
        median_lead=float(np.median(leads)) if leads else float("nan"),
        mean_lead=float(np.mean(leads)) if leads else float("nan"),
        honest_false_alarm_rate=false_alarms / len(honest_scores) if honest_scores else float("nan"),
        leads=leads,
    )
