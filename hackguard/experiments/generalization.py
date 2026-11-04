"""Does a probe trained on one kind of cheating detect a different kind? (spec Step 6)

Two families of hack:
- test edits:  editing or weakening the test file, or test configuration tricks
- code hacks:  special-casing test inputs, hardcoding outputs, exiting early

We train on honest samples plus hacks of one family only (train split), and test on honest
samples plus hacks of the *other* family (test split), in both directions. Trajectories
that contain both families are left out, so each side is a clean example of one family.
High AUROC here suggests the probe reads a general "about to cheat" signal rather than
something specific to one trick.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from hackguard.activations.extract import load_layer, load_samples
from hackguard.probes.metrics import auroc
from hackguard.probes.splits import sample_splits
from hackguard.probes.train import C_GRID, train_logreg
from hackguard.trajectories.labeling import CODE_HACK_TYPES, TEST_EDIT_TYPES

FAMILIES = {"test_edits": TEST_EDIT_TYPES, "code_hacks": CODE_HACK_TYPES}


def family_of(hack_types: str) -> str | None:
    """'test_edits', 'code_hacks', or None if a hack uses both families (or is not a hack)."""
    types = set(hack_types.split())
    in_families = [name for name, members in FAMILIES.items() if types & members]
    return in_families[0] if len(in_families) == 1 else None


def run_generalization(act_dir: Path, results_dir: Path, layer: int, seed: int) -> dict[str, Any]:
    samples = load_samples(act_dir)
    x = load_layer(act_dir, layer)
    split = samples["problem_id"].map(sample_splits(samples, seed)).to_numpy()
    family = samples["hack_types"].map(family_of).to_numpy()
    y = samples["label"].to_numpy()
    honest = y == 0

    report: dict[str, Any] = {"layer": layer}
    for train_family, test_family in (("test_edits", "code_hacks"), ("code_hacks", "test_edits")):
        train_mask = (split == "train") & (honest | (family == train_family))
        test_mask = (split == "test") & (honest | (family == test_family))
        same_mask = (split == "test") & (honest | (family == train_family))  # in-family reference
        key = f"train_{train_family}__test_{test_family}"
        if len(np.unique(y[train_mask])) < 2 or len(np.unique(y[test_mask])) < 2:
            report[key] = {"skipped": "not enough hacks of one family"}
            continue
        probe = train_logreg(x[train_mask], y[train_mask], layer, C_GRID[1])
        report[key] = {
            "auroc_other_family": auroc(y[test_mask], probe.score(x[test_mask])),
            "auroc_same_family": auroc(y[same_mask], probe.score(x[same_mask])),
            "train_hack_samples": int((y[train_mask] == 1).sum()),
            "test_hack_samples": int((y[test_mask] == 1).sum()),
        }
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "generalization.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def family_counts(samples: pd.DataFrame) -> dict[str, int]:
    per_traj = samples.groupby("trajectory_id")["hack_types"].first().map(family_of)
    return {str(k): int(v) for k, v in per_traj.value_counts(dropna=False).items()}
