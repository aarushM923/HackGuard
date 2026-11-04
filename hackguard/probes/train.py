"""Train linear probes on saved activations, one per layer (spec Step 4.1 and 4.2).

A linear probe is the simplest possible reader of the model's internals: a weighted sum of
the activation's numbers plus a bias, squashed into a probability. If such a simple reader
can predict cheating, the information must be plainly present in the residual stream.

Two kinds, each trained separately for every saved layer:
- Logistic regression: learns the weights that best separate hack from honest samples.
  We standardize each number first (subtract its mean, divide by its spread) so all
  dimensions are on the same scale, and choose the regularization strength C on the
  validation split. class_weight="balanced" stops the probe from ignoring the rarer class.
- Difference in means: the direction from the average honest activation to the average
  hack activation; a sample's score is how far it points along that direction. No training
  loop at all, so it is a useful sanity baseline.

The layer used for headline numbers is chosen by validation AUROC, never by test AUROC.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from hackguard.activations.extract import load_layer, load_meta, load_samples
from hackguard.probes.metrics import auroc, bootstrap_auroc_ci, precision_at_fpr
from hackguard.probes.splits import sample_splits

C_GRID = (0.001, 0.01, 0.1)
MAX_TRAIN_SAMPLES = 50_000  # cap for speed; windows from one trajectory are very similar anyway


@dataclass
class LinearProbe:
    """score(x) = ((x - mean) / scale) . weight + bias; probability = sigmoid(score)."""

    kind: str  # "logreg" or "diffmeans"
    layer: int
    mean: np.ndarray
    scale: np.ndarray
    weight: np.ndarray
    bias: float

    def score(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(((x - self.mean) / self.scale) @ self.weight + self.bias)

    def prob(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(1.0 / (1.0 + np.exp(-self.score(x))))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, kind=self.kind, layer=self.layer, mean=self.mean, scale=self.scale,
                 weight=self.weight, bias=self.bias)

    @classmethod
    def load(cls, path: Path) -> "LinearProbe":
        d = np.load(path)
        return cls(str(d["kind"]), int(d["layer"]), d["mean"], d["scale"], d["weight"], float(d["bias"]))


def train_logreg(x: np.ndarray, y: np.ndarray, layer: int, c: float) -> LinearProbe:
    mean = x.mean(axis=0)
    scale = x.std(axis=0) + 1e-6  # tiny constant avoids dividing by zero for constant dimensions
    model = LogisticRegression(C=c, class_weight="balanced", max_iter=2000)
    model.fit((x - mean) / scale, y)
    return LinearProbe("logreg", layer, mean, scale, model.coef_[0].astype(np.float32), float(model.intercept_[0]))


def train_diffmeans(x: np.ndarray, y: np.ndarray, layer: int) -> LinearProbe:
    hack_mean, honest_mean = x[y == 1].mean(axis=0), x[y == 0].mean(axis=0)
    direction = hack_mean - honest_mean
    midpoint = (hack_mean + honest_mean) / 2
    # Score = how far past the midpoint the sample lies along the direction.
    return LinearProbe("diffmeans", layer, np.zeros_like(direction), np.ones_like(direction),
                       direction.astype(np.float32), float(-(midpoint @ direction)))


def evaluate(probe: LinearProbe, x: np.ndarray, samples: pd.DataFrame) -> dict[str, float]:
    """All test metrics for one probe on one set of samples."""
    y = samples["label"].to_numpy()
    s = probe.score(x)
    lo, hi = bootstrap_auroc_ci(y, s, samples["problem_id"].to_numpy())
    # Trajectory level: does the highest score in each trajectory's window separate them?
    per_traj = pd.DataFrame({"t": samples["trajectory_id"].to_numpy(), "y": y, "s": s}).groupby("t").agg({"y": "first", "s": "max"})
    # The control from the spec's risk table: among impossible tasks only, hack vs honest
    # failure. Both see an impossible task, so a probe that only detects "this task is
    # impossible" would score near 0.5 here.
    impossible = (samples["variant"] == "impossible").to_numpy()
    return {
        "auroc": auroc(y, s),
        "auroc_ci_low": lo,
        "auroc_ci_high": hi,
        "precision_at_5pct_fpr": precision_at_fpr(y, s, 0.05),
        "trajectory_auroc": auroc(per_traj["y"].to_numpy(), per_traj["s"].to_numpy()),
        "impossible_only_auroc": auroc(y[impossible], s[impossible]),
        "n_samples": int(len(y)),
        "n_hack_trajectories": int(per_traj["y"].sum()),
        "n_honest_trajectories": int((per_traj["y"] == 0).sum()),
    }


def split_masks(samples: pd.DataFrame, seed: int) -> dict[str, np.ndarray]:
    split = sample_splits(samples, seed)
    names = samples["problem_id"].map(split).to_numpy()
    return {name: names == name for name in ("train", "val", "test")}


def train_all_layers(act_dir: Path, probe_dir: Path, results_dir: Path, seed: int = 0) -> pd.DataFrame:
    """Train and evaluate both probe kinds at every saved layer. Returns the results table."""
    meta = load_meta(act_dir)
    samples = load_samples(act_dir)
    masks = split_masks(samples, seed)
    y = samples["label"].to_numpy()
    for name, mask in masks.items():
        if len(np.unique(y[mask])) < 2:
            raise ValueError(f"the {name} split has only one class; generate more trajectories (need hacks and honest runs)")

    rng = np.random.default_rng(seed)
    train_idx = np.where(masks["train"])[0]
    if len(train_idx) > MAX_TRAIN_SAMPLES:
        train_idx = np.sort(rng.choice(train_idx, MAX_TRAIN_SAMPLES, replace=False))

    rows = []
    for layer in meta["layers"]:
        x = load_layer(act_dir, layer)
        x_train, y_train = x[train_idx], y[train_idx]
        x_val, y_val = x[masks["val"]], y[masks["val"]]

        # Choose C by validation AUROC.
        best_c, best_val, best_probe = C_GRID[0], -1.0, None
        for c in C_GRID:
            probe = train_logreg(x_train, y_train, layer, c)
            val = auroc(y_val, probe.score(x_val))
            if val > best_val:
                best_c, best_val, best_probe = c, val, probe
        assert best_probe is not None
        diff = train_diffmeans(x_train, y_train, layer)

        for probe, val_auroc in ((best_probe, best_val), (diff, auroc(y_val, diff.score(x_val)))):
            probe.save(probe_dir / f"{probe.kind}_layer{layer:02d}.npz")
            metrics = evaluate(probe, x[masks["test"]], samples[masks["test"]].reset_index(drop=True))
            rows.append({"probe": probe.kind, "layer": layer, "val_auroc": val_auroc,
                         "C": best_c if probe.kind == "logreg" else None, **metrics})
        print(f"  layer {layer:2d}: logreg test AUROC {rows[-2]['auroc']:.3f}, diff-in-means {rows[-1]['auroc']:.3f}")

    table = pd.DataFrame(rows)
    results_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(results_dir / "probe_layers.csv", index=False)
    logreg = table[table["probe"] == "logreg"]
    best = logreg.loc[logreg["val_auroc"].idxmax()]
    (results_dir / "best_layer.json").write_text(json.dumps({"layer": int(best["layer"]), "chosen_by": "validation AUROC (logreg)"}, indent=2), encoding="utf-8")
    return table


def best_layer(results_dir: Path) -> int:
    return int(json.loads((results_dir / "best_layer.json").read_text(encoding="utf-8"))["layer"])
