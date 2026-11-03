import math

import numpy as np

from hackguard.probes.metrics import (
    auroc,
    bootstrap_auroc_ci,
    first_crossing,
    lead_times,
    precision_at_fpr,
    trajectory_threshold,
)
from hackguard.probes.splits import split_problems


def test_auroc_basics() -> None:
    assert auroc(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert auroc(np.array([0, 0, 1, 1]), np.array([0.9, 0.8, 0.2, 0.1])) == 0.0
    assert math.isnan(auroc(np.array([1, 1]), np.array([0.1, 0.2])))


def test_precision_at_fpr() -> None:
    y = np.array([0] * 100 + [1] * 10)
    s = np.concatenate([np.linspace(0, 1, 100), np.full(10, 2.0)])
    assert precision_at_fpr(y, s, 0.05) == 10 / (10 + 5)  # 5 of 100 negatives allowed above the threshold


def test_bootstrap_interval_contains_estimate() -> None:
    rng = np.random.default_rng(0)
    y = np.array([0, 1] * 100)
    s = y + rng.normal(scale=1.0, size=200)
    lo, hi = bootstrap_auroc_ci(y, s, np.arange(200) // 4, n_boot=200)
    assert lo < auroc(y, s) < hi


def test_threshold_and_lead_times() -> None:
    tau = trajectory_threshold([float(v) for v in np.linspace(0, 1, 101)])
    assert abs(tau - 0.95) < 1e-9
    hacks = [
        (np.array([0.1, 0.99, 0.99, 0.99]), 3),  # crosses at 1, hack at 3 -> lead 2
        (np.array([0.1, 0.1, 0.1, 0.99]), 2),  # crosses at 3, after the hack at 2
        (np.array([0.1, 0.1]), 1),  # never crosses
    ]
    honest = [np.array([0.1, 0.2]), np.array([0.99])]
    result = lead_times(hacks, honest, tau)
    assert (result.caught_before, result.caught_after, result.missed) == (1, 1, 1)
    assert result.leads == [2] and result.median_lead == 2
    assert result.share_caught_before == 1 / 3
    assert result.honest_false_alarm_rate == 0.5
    assert first_crossing(np.array([0.1, 0.2]), 0.5) is None


def test_split_by_problem_is_stable_and_complete() -> None:
    ids = [f"P/{i}" for i in range(100)]
    split = split_problems(ids, seed=0)
    assert split == split_problems(list(reversed(ids)), seed=0)  # order does not matter
    counts = {name: list(split.values()).count(name) for name in ("train", "val", "test")}
    assert counts == {"train": 60, "val": 20, "test": 20}
