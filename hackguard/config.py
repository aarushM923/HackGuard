"""Settings shared across the pipeline: which models, which layers, and where files go.

Two presets exist:
- "laptop": a tiny model (Qwen2.5-Coder-0.5B) on about 20 tasks, run with plain
  transformers on the Mac GPU. It exists to find bugs for free before renting a GPU.
- "gpu": the real run on the EC2 g5.2xlarge (one 24 GB A10G GPU) with the 7B model,
  vLLM for fast generation, and Docker for running model-written code.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

MAIN_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"
TINY_MODEL = "Qwen/Qwen2.5-Coder-0.5B-Instruct"

# Second model for the cross-model transfer experiment: a 6.7B coding model from a different
# family (Llama-style architecture, different tokenizer, 32 layers of width 4,096). It is
# not "gated", so it needs no Hugging Face account or license approval. In fp16 it takes
# about 13.5 GB, which fits the 24 GB GPU. (Llama-3.1-8B-Instruct would also fit but needs
# Meta's license approval and an HF_TOKEN.)
SECOND_MODEL = "deepseek-ai/deepseek-coder-6.7b-instruct"
# For the laptop test of the transfer code: a small ungated model with a *different*
# tokenizer from Qwen, so the laptop run exercises the cross-tokenizer position mapping.
TINY_SECOND_MODEL = "HuggingFaceTB/SmolLM2-360M-Instruct"

PROMPT_NAMES = ("neutral", "pressure", "no_modify")

DEFAULT_RUNS_DIR = Path("runs")
TASK_SUITE_PATH = Path("data/task_suite/tasks.jsonl")


def probe_layers(num_layers: int, every: int = 4) -> list[int]:
    """Which layers we read: every 4th one, as in the spec.

    Layer L means "the residual stream right after transformer block L" (blocks counted
    from 1). For a 28-block model this gives [4, 8, 12, 16, 20, 24, 28].
    """
    return list(range(every, num_layers + 1, every))


def model_slug(model_id: str) -> str:
    """A folder-safe short name for a model, e.g. 'Qwen2.5-Coder-7B-Instruct'."""
    return model_id.split("/")[-1]


@dataclass
class RunConfig:
    """Every setting for one end-to-end run. Saved as config.json in the run folder."""

    name: str
    model: str
    second_model: str
    backend: str  # "hf" (plain transformers) or "vllm"
    runner: str  # "local" or "docker": where model-written code runs
    num_problems: int | None  # None = all problems in the task suite
    prompts: list[str] = field(default_factory=lambda: list(PROMPT_NAMES))
    seeds: list[int] = field(default_factory=lambda: [0])
    max_turns: int = 8
    max_new_tokens: int = 1024  # per agent turn
    max_context_tokens: int = 8192
    temperature: float = 0.7
    generation_batch_size: int = 16  # only used by the "hf" backend
    window: int = 64  # tokens before the hack (or matched position) saved per trajectory
    steering_tasks_per_variant: int = 50
    steering_strengths: list[float] = field(default_factory=lambda: [0.0, 4.0, 8.0])
    judge_max_items: int = 400
    seed: int = 0  # for sampling and data splits

    @property
    def run_dir(self) -> Path:
        return DEFAULT_RUNS_DIR / self.name

    def save(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "config.json").write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, run_dir: Path) -> "RunConfig":
        return cls(**json.loads((run_dir / "config.json").read_text(encoding="utf-8")))


def laptop_config(name: str = "laptop") -> RunConfig:
    """Tiny end-to-end test: 30 problems x 3 variants = 90 tasks, small model, Mac GPU.

    30 problems: with 7 or 14, the problem-level split left the test split
    without a single hack (the tiny model rarely cheats), so no probe could be evaluated."""
    return RunConfig(
        name=name,
        model=TINY_MODEL,
        second_model=TINY_SECOND_MODEL,
        backend="hf",
        runner="local",
        num_problems=30,
        max_turns=5,
        max_new_tokens=512,
        max_context_tokens=6144,
        generation_batch_size=8,
        window=32,
        steering_tasks_per_variant=4,
        steering_strengths=[0.0, 8.0],
        judge_max_items=10,
    )


def gpu_config(name: str = "gpu") -> RunConfig:
    """The real run on one A10G: all 400 problems x 3 variants x 3 prompts."""
    return RunConfig(
        name=name,
        model=MAIN_MODEL,
        second_model=SECOND_MODEL,
        backend="vllm",
        runner="docker",
        num_problems=None,
    )


def pilot_config(name: str = "pilot") -> RunConfig:
    """The GPU setup on 34 problems (102 tasks): checks speed and how often the model
    cheats before committing to the full run, as the spec suggests."""
    cfg = gpu_config(name)
    cfg.num_problems = 34
    return cfg


PRESETS = {"laptop": laptop_config, "pilot": pilot_config, "gpu": gpu_config}
