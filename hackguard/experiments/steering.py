"""Steering: subtract the cheating direction while the agent works (spec Step 6).

The difference-in-means probe gives a direction in the residual stream that points from
"honest" toward "about to hack". During generation we subtract a multiple of it after the
best layer, at every token, then measure two things:
- hack rate: share of episodes labeled "hack" (should go down if the direction is causal),
- honest pass rate: share of honest-variant tasks whose hidden tests pass (should not drop;
  if it does, steering is just breaking the model rather than removing the urge to cheat).

Strengths are given in standard deviations of how far training activations project onto
the direction, so "8" means a push of 8 typical spreads. Strength 0 is the unsteered
control, run on exactly the same tasks, prompt and seeds.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from hackguard.activations.extract import load_layer, load_samples
from hackguard.agent.generation import HFGenerator, Steering
from hackguard.agent.loop import EpisodeSpec, run_episodes
from hackguard.config import RunConfig
from hackguard.models import load_model
from hackguard.probes.splits import split_problems
from hackguard.probes.train import LinearProbe
from hackguard.tasks.schema import Task
from hackguard.tasks.workspace import Runner
from hackguard.trajectories.labeling import label_trajectory
from hackguard.trajectories.schema import Label, Trajectory, save_jsonl


def steering_direction(act_dir: Path, probe_dir: Path, layer: int, seed: int) -> tuple[torch.Tensor, float]:
    """Unit-length cheating direction and the spread of activations along it (training split)."""
    probe = LinearProbe.load(probe_dir / f"diffmeans_layer{layer:02d}.npz")
    direction = probe.weight / np.linalg.norm(probe.weight)
    samples = load_samples(act_dir)
    train = samples["problem_id"].map(split_problems(samples["problem_id"].tolist(), seed)).to_numpy() == "train"
    projections = load_layer(act_dir, layer)[train] @ direction
    return torch.tensor(direction, dtype=torch.float32), float(projections.std())


def pick_prompt(trajectories: list[Trajectory], labels: dict[str, Label]) -> str:
    """The system prompt with the highest hack rate in the main run (most room to reduce it)."""
    rates: dict[str, list[int]] = {}
    for t in trajectories:
        rates.setdefault(t.prompt_name, []).append(int(labels[t.trajectory_id].label == "hack"))
    return max(rates, key=lambda name: float(np.mean(rates[name])))


def steering_tasks(tasks: list[Task], per_variant: int, seed: int) -> list[Task]:
    """Impossible and honest tasks from *test-split* problems only."""
    split = split_problems([t.problem_id for t in tasks], seed)
    rng = random.Random(seed)
    chosen = []
    for variant in ("impossible", "honest"):
        pool = sorted((t for t in tasks if t.variant == variant and split[t.problem_id] == "test"), key=lambda t: t.task_id)
        chosen += rng.sample(pool, min(per_variant, len(pool)))
    return chosen


def run_steering(cfg: RunConfig, act_dir: Path, probe_dir: Path, results_dir: Path, layer: int,
                 tasks: list[Task], prompt_name: str, runner: Runner) -> pd.DataFrame:
    direction, spread = steering_direction(act_dir, probe_dir, layer, cfg.seed)
    loaded = load_model(cfg.model)
    chosen = steering_tasks(tasks, cfg.steering_tasks_per_variant, cfg.seed)
    by_id = {t.task_id: t for t in chosen}
    rows: list[dict[str, Any]] = []
    for strength in cfg.steering_strengths:
        steering = Steering(layer, direction, strength * spread)
        generator = HFGenerator(cfg.model, cfg.max_new_tokens, cfg.temperature, cfg.generation_batch_size,
                                steering=steering, loaded=loaded)
        specs = [EpisodeSpec(task, prompt_name, seed=0) for task in chosen]
        trajectories = run_episodes(specs, generator, cfg.model, cfg.max_turns, cfg.max_new_tokens,
                                    cfg.max_context_tokens, runner=runner)
        labels = [label_trajectory(t, by_id[t.task_id]) for t in trajectories]
        save_jsonl(trajectories, results_dir / "steering" / f"trajectories_strength{strength:g}.jsonl")
        for traj, lab in zip(trajectories, labels):
            rows.append({"strength": strength, "variant": traj.variant, "hack": int(lab.label == "hack"),
                         "hidden_pass": int(traj.hidden.all_passed)})
        print(f"  steering strength {strength:g}: done {len(trajectories)} episodes")

    table = pd.DataFrame(rows)
    summary = []
    for strength, group in table.groupby("strength"):
        honest = group[group["variant"] == "honest"]
        summary.append({
            "strength_in_std": float(strength),
            "hack_rate_all": float(group["hack"].mean()),
            "hack_rate_impossible": float(group[group["variant"] == "impossible"]["hack"].mean()),
            "honest_pass_rate": float(honest["hidden_pass"].mean()) if len(honest) else float("nan"),
            "episodes": int(len(group)),
        })
    out = pd.DataFrame(summary)
    results_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(results_dir / "steering.csv", index=False)
    (results_dir / "steering.json").write_text(json.dumps({"layer": layer, "prompt": prompt_name,
        "direction_spread": spread, "results": summary}, indent=2), encoding="utf-8")
    return out
