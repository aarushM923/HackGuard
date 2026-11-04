"""Split by problem, not by trajectory (spec Step 4.4).

All trajectories of one problem (every variant, prompt and seed) land in the same split.
Otherwise the probe could memorize something about a specific problem from training
trajectories and recognize it again at test time, which would overstate how well it works.

60% of problems train the probes, 20% (validation) choose settings such as the layer and
the alarm threshold, and 20% (test) are only used for the final numbers.

The split is *stratified*: problems where the model cheated at least once are shuffled and
cut 60/20/20 on their own, and so are the other problems. Hacks are rare, so a plain random
cut can leave the validation or test split with no hacks at all, and then nothing can be
measured there. Stratifying keeps the share of hack problems about equal in every split.

Every step builds the split from the main model's samples.csv with `sample_splits`, so all
steps (training, lead time, baselines, experiments) agree on which problem is where.
"""

from __future__ import annotations

import random

import pandas as pd

SPLITS = ("train", "val", "test")


def _cut(ids: list[str], rng: random.Random, fractions: tuple[float, float]) -> dict[str, str]:
    rng.shuffle(ids)
    n_train = round(len(ids) * fractions[0])
    n_val = round(len(ids) * fractions[1])
    if len(ids) >= 3:  # with 3 or more problems, make sure no split is empty
        n_train = min(max(n_train, 1), len(ids) - 2)
        n_val = min(max(n_val, 1), len(ids) - n_train - 1)
    return {pid: "train" if i < n_train else "val" if i < n_train + n_val else "test" for i, pid in enumerate(ids)}


def split_problems(
    problem_ids: list[str],
    seed: int = 0,
    fractions: tuple[float, float] = (0.6, 0.2),
    hack_problem_ids: set[str] | None = None,
) -> dict[str, str]:
    """Map each problem id to "train", "val" or "test".

    If `hack_problem_ids` is given, those problems and the rest are split separately
    (stratified), so every split gets its share of problems with hacks.
    """
    ids = sorted(set(problem_ids))
    rng = random.Random(seed)
    if hack_problem_ids is None:
        return _cut(ids, rng, fractions)
    hacky = [pid for pid in ids if pid in hack_problem_ids]
    clean = [pid for pid in ids if pid not in hack_problem_ids]
    return {**_cut(hacky, rng, fractions), **_cut(clean, rng, fractions)}


def sample_splits(samples: pd.DataFrame, seed: int = 0) -> dict[str, str]:
    """The run's problem split, from a samples table (label 1 = sample from a hack trajectory)."""
    hack_ids = set(samples.loc[samples["label"] == 1, "problem_id"])
    return split_problems(samples["problem_id"].tolist(), seed, hack_problem_ids=hack_ids)
