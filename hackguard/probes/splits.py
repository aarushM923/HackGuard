"""Split by problem, not by trajectory (spec Step 4.4).

All trajectories of one problem (every variant, prompt and seed) land in the same split.
Otherwise the probe could memorize something about a specific problem from training
trajectories and recognize it again at test time, which would overstate how well it works.

60% of problems train the probes, 20% (validation) choose settings such as the layer and
the alarm threshold, and 20% (test) are only used for the final numbers.
"""

from __future__ import annotations

import random

SPLITS = ("train", "val", "test")


def split_problems(problem_ids: list[str], seed: int = 0, fractions: tuple[float, float] = (0.6, 0.2)) -> dict[str, str]:
    """Map each problem id to "train", "val" or "test"."""
    ids = sorted(set(problem_ids))
    random.Random(seed).shuffle(ids)
    n_train = round(len(ids) * fractions[0])
    n_val = round(len(ids) * fractions[1])
    result = {}
    for i, pid in enumerate(ids):
        result[pid] = "train" if i < n_train else "val" if i < n_train + n_val else "test"
    return result
