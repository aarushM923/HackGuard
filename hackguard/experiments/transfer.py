"""Cross-model transfer: does a probe trained on Qwen work on a second model? (spec Step 6)

The two models have different sizes of residual stream (Qwen 7B: 3,584 numbers per token;
Llama 8B: 4,096), so Qwen's probe cannot be applied to Llama directly. Instead:
1. Run the second model over the *same* trajectories and save its activations at the same
   (message, character) positions as Qwen's samples (extract.py, positions_from=...).
2. Fit a linear map from the second model's activations to Qwen's (ridge regression, on
   training-split samples only; the labels are not used to fit it).
3. Map the second model's test activations into "Qwen space" and apply Qwen's probe.

For reference we also report a probe trained directly on the second model (an upper bound
for what its activations contain) and Qwen's own probe on the same test samples.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

from hackguard.activations.extract import load_layer, load_meta, load_samples
from hackguard.probes.metrics import auroc
from hackguard.probes.splits import split_problems
from hackguard.probes.train import C_GRID, LinearProbe, train_logreg

RIDGE_ALPHAS = (1.0, 10.0, 100.0, 1000.0)
MAX_ALIGN_SAMPLES = 30_000
KEY = ["trajectory_id", "tokens_before"]  # identifies the same sample in both models


def matching_layer(layer_a: int, blocks_a: int, layers_b: list[int], blocks_b: int) -> int:
    """The second model's saved layer at the most similar relative depth."""
    depth = layer_a / blocks_a
    return min(layers_b, key=lambda lb: abs(lb / blocks_b - depth))


def run_transfer(act_dir_a: Path, act_dir_b: Path, probe_dir_a: Path, results_dir: Path,
                 layer_a: int, seed: int) -> dict[str, Any]:
    meta_a, meta_b = load_meta(act_dir_a), load_meta(act_dir_b)
    layer_b = matching_layer(layer_a, meta_a["num_blocks"], meta_b["layers"], meta_b["num_blocks"])

    # Line up rows that are the same sample in both models.
    sa = load_samples(act_dir_a).reset_index().rename(columns={"index": "row_a"})
    sb = load_samples(act_dir_b).reset_index().rename(columns={"index": "row_b"})
    both = sa.merge(sb[KEY + ["row_b"]], on=KEY)
    xa = load_layer(act_dir_a, layer_a)[both["row_a"].to_numpy()]
    xb = load_layer(act_dir_b, layer_b)[both["row_b"].to_numpy()]
    y = both["label"].to_numpy()
    split = both["problem_id"].map(split_problems(both["problem_id"].tolist(), seed)).to_numpy()
    train, val, test = split == "train", split == "val", split == "test"

    rng = np.random.default_rng(seed)
    fit_idx = np.where(train)[0]
    if len(fit_idx) > MAX_ALIGN_SAMPLES:
        fit_idx = rng.choice(fit_idx, MAX_ALIGN_SAMPLES, replace=False)

    # Choose the ridge strength by how well the map predicts Qwen activations on validation.
    best_alpha, best_r2, best_map = RIDGE_ALPHAS[0], -np.inf, None
    for alpha in RIDGE_ALPHAS:
        mapping = Ridge(alpha=alpha).fit(xb[fit_idx], xa[fit_idx])
        r2 = r2_score(xa[val], mapping.predict(xb[val]))
        if r2 > best_r2:
            best_alpha, best_r2, best_map = alpha, r2, mapping
    assert best_map is not None

    probe_a = LinearProbe.load(probe_dir_a / f"logreg_layer{layer_a:02d}.npz")
    native_b = train_logreg(xb[train], y[train], layer_b, C_GRID[1])
    mapped_test = best_map.predict(xb[test])
    report = {
        "model_a": meta_a["model"], "model_b": meta_b["model"],
        "layer_a": layer_a, "layer_b": layer_b,
        "ridge_alpha": best_alpha,
        "alignment_r2_test": float(r2_score(xa[test], mapped_test)),
        "auroc_qwen_probe_on_mapped_b": auroc(y[test], probe_a.score(mapped_test)),
        "auroc_native_b_probe": auroc(y[test], native_b.score(xb[test])),
        "auroc_qwen_probe_on_qwen": auroc(y[test], probe_a.score(xa[test])),
        "n_test_samples": int(test.sum()),
        "n_matched_samples": int(len(both)),
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "transfer.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
