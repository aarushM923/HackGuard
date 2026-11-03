"""Generate trajectories for a run: every chosen task x every system prompt x every seed.

Results are appended to runs/<name>/trajectories.jsonl one chunk at a time. If the run
stops halfway (crash, instance stopped), running the same command again skips every
trajectory already on disk instead of paying to generate it again.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

from hackguard.agent.generation import Generator
from hackguard.agent.loop import EpisodeSpec, run_episodes
from hackguard.config import TASK_SUITE_PATH, RunConfig
from hackguard.tasks.schema import Task, load_tasks
from hackguard.tasks.workspace import LOCAL_RUNNER, DockerRunner, Runner
from hackguard.trajectories.schema import save_jsonl


def make_runner(name: str) -> Runner:
    if name == "docker":
        return DockerRunner()
    if name == "local":
        return LOCAL_RUNNER
    raise ValueError(f"unknown runner {name!r}")


def select_tasks(tasks: list[Task], num_problems: int | None, seed: int) -> list[Task]:
    """Keep all three variants of `num_problems` randomly chosen problems (all if None)."""
    problem_ids = sorted({t.problem_id for t in tasks})
    if num_problems is not None and num_problems < len(problem_ids):
        problem_ids = sorted(random.Random(seed).sample(problem_ids, num_problems))
    chosen = set(problem_ids)
    return [t for t in tasks if t.problem_id in chosen]


def episode_specs(tasks: list[Task], prompts: list[str], seeds: list[int]) -> list[EpisodeSpec]:
    return [EpisodeSpec(task, prompt, seed) for task in tasks for prompt in prompts for seed in seeds]


def existing_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open(encoding="utf-8") as f:
        return {json.loads(line)["trajectory_id"] for line in f if line.strip()}


def generate_trajectories(
    specs: list[EpisodeSpec],
    generator: Generator,
    cfg: RunConfig,
    out_path: Path,
    runner: Runner,
    chunk_size: int = 256,
) -> int:
    """Run all specs not already in out_path, appending results. Returns how many ran."""
    done = existing_ids(out_path)
    todo = [s for s in specs if s.trajectory_id not in done]
    print(f"{len(done)} trajectories already saved, {len(todo)} to generate.")
    start = time.time()
    for i in range(0, len(todo), chunk_size):
        chunk = todo[i : i + chunk_size]
        trajectories = run_episodes(
            chunk, generator, cfg.model, cfg.max_turns, cfg.max_new_tokens, cfg.max_context_tokens, runner=runner
        )
        save_jsonl(trajectories, out_path, append=True)
        finished = i + len(chunk)
        rate = finished / max(time.time() - start, 1e-9)
        print(f"  {finished}/{len(todo)} episodes done ({rate * 60:.1f} per minute)")
    return len(todo)


def run_generation(cfg: RunConfig, generator: Generator, task_path: Path = TASK_SUITE_PATH) -> Path:
    tasks = select_tasks(load_tasks(task_path), cfg.num_problems, cfg.seed)
    specs = episode_specs(tasks, cfg.prompts, cfg.seeds)
    out_path = cfg.run_dir / "trajectories.jsonl"
    print(f"{len(tasks)} tasks x {len(cfg.prompts)} prompts x {len(cfg.seeds)} seeds = {len(specs)} episodes")
    generate_trajectories(specs, generator, cfg, out_path, make_runner(cfg.runner))
    return out_path
