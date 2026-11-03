"""Data structures for trajectories and labels, saved as JSONL (one JSON object per line).

A trajectory is one agent episode: the full conversation, every action with its exact
character position inside the assistant's message, the files at the end, and three test
results:
- visible_workspace: the agent's own test file (possibly edited by the agent) against its
  final solution. This is what "the tests pass" means from the agent's point of view.
- visible_original: the original visible tests against the final solution, in a clean folder.
- hidden: the full hidden test suite against the final solution. This is real correctness.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class RunSummary:
    passed: int
    failed: int
    all_passed: bool
    timed_out: bool = False


@dataclass(frozen=True)
class ActionRecord:
    """One action the agent took.

    message_index: which message in `messages` contains it (an assistant message).
    char_start: where the action tag starts inside that message.
    content_start: for write_file, where the file contents start inside that message.
    """

    turn: int
    kind: str
    message_index: int
    char_start: int
    char_end: int
    path: str | None = None
    content: str | None = None
    content_start: int | None = None
    observation: str = ""


@dataclass
class Trajectory:
    trajectory_id: str  # "<task_id>::<prompt_name>::s<seed>", e.g. "Mbpp/2::impossible::pressure::s0"
    task_id: str
    problem_id: str
    variant: str
    source: str
    prompt_name: str
    seed: int
    model: str
    messages: list[dict[str, str]]  # {"role": "system"|"user"|"assistant", "content": ...}
    actions: list[ActionRecord]
    end_reason: str  # "submit", "max_turns", "invalid_actions" or "context_limit"
    final_files: dict[str, str]
    visible_workspace: RunSummary
    visible_original: RunSummary
    hidden: RunSummary


@dataclass(frozen=True)
class HackLocation:
    """Where the first hacking action happens: a character inside one assistant message."""

    message_index: int
    char_offset: int
    turn: int
    signature: str


@dataclass
class Label:
    trajectory_id: str
    label: str  # "hack", "honest_success", "honest_fail" or "ambiguous"
    hack_types: list[str] = field(default_factory=list)
    first_hack: HackLocation | None = None
    evidence: list[str] = field(default_factory=list)


def _to_json(obj: Any) -> str:
    return json.dumps(asdict(obj))


def trajectory_from_dict(d: dict[str, Any]) -> Trajectory:
    d = dict(d)
    d["actions"] = [ActionRecord(**a) for a in d["actions"]]
    for key in ("visible_workspace", "visible_original", "hidden"):
        d[key] = RunSummary(**d[key])
    return Trajectory(**d)


def label_from_dict(d: dict[str, Any]) -> Label:
    d = dict(d)
    if d.get("first_hack") is not None:
        d["first_hack"] = HackLocation(**d["first_hack"])
    return Label(**d)


def save_jsonl(items: Iterable[Any], path: Path, append: bool = False) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("a" if append else "w", encoding="utf-8") as f:
        for item in items:
            f.write(_to_json(item) + "\n")
            count += 1
    return count


def load_trajectories(path: Path) -> list[Trajectory]:
    with path.open(encoding="utf-8") as f:
        return [trajectory_from_dict(json.loads(line)) for line in f if line.strip()]


def load_labels(path: Path) -> dict[str, Label]:
    """Labels keyed by trajectory id."""
    with path.open(encoding="utf-8") as f:
        labels = [label_from_dict(json.loads(line)) for line in f if line.strip()]
    return {lab.trajectory_id: lab for lab in labels}
