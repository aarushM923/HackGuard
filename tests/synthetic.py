"""Builds a fake activations folder for testing Steps 4 to 6 without running a model.

Hack samples are ordinary random vectors pushed along one hidden direction, at every layer
but more strongly at deeper layers, so a working probe must find it and should prefer
deeper layers. Problems are numbered so the train/val/test split has all classes.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from hackguard.activations.extract import ShardWriter

HIDDEN = 16
LAYERS = [4, 8]


def make_activation_dir(root: Path, n_problems: int = 40, per_traj: int = 4, seed: int = 0,
                        hidden: int = HIDDEN, layers: list[int] = LAYERS, num_blocks: int = 8,
                        name: str = "fake-model") -> tuple[Path, np.ndarray]:
    rng = np.random.default_rng(seed)
    direction = rng.normal(size=hidden)
    direction /= np.linalg.norm(direction)
    out = root / name
    writer = ShardWriter(out)
    for p in range(n_problems):
        for variant, is_hack, types in (("impossible", 1, "edit_tests" if p % 2 else "special_case"),
                                        ("honest", 0, ""), ("impossible", 0, "")):
            tid = f"P/{p}::{variant}::{is_hack}"
            acts = rng.normal(size=(per_traj, len(layers), hidden))
            if is_hack:
                for li in range(len(layers)):
                    acts[:, li, :] += (1.0 + 2.0 * li) * direction
            samples = [{
                "trajectory_id": tid, "problem_id": f"P/{p}", "variant": variant, "prompt_name": "neutral",
                "label": is_hack, "hack_types": types, "token_index": 10 + k, "message_index": 2,
                "char_end": 5 * (k + 1), "tokens_before": per_traj - k,
            } for k in range(per_traj)]
            writer.add(acts.astype(np.float16), samples)
    writer.flush()
    import pandas as pd
    pd.DataFrame(writer.rows).to_csv(out / "samples.csv", index=False)
    (out / "meta.json").write_text(json.dumps({"model": name, "layers": layers, "num_blocks": num_blocks,
                                               "hidden_size": hidden, "samples": writer.total}), encoding="utf-8")
    return out, direction
