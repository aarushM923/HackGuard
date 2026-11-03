"""Data structures for the task suite, and saving/loading them as JSONL.

Vocabulary used across the project:
- A *problem* is one coding exercise from HumanEval+ or MBPP+ (e.g. "write has_close_elements").
- A *task* is one problem turned into a specific variant (honest, impossible or tempting)
  that an agent will be asked to solve.
- A *test case* is one (input, expected output) pair.

Test values (inputs and expected outputs) are stored as Python source text, for example
"[(1, 2), 'abc']", rather than as JSON. JSON cannot tell a tuple from a list and has no
sets, so storing Python text keeps the values exact. The text is turned back into values
with ast.literal_eval, which only reads plain data and never runs code.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal

# The three kinds of task. "Literal" tells the type checker only these strings are allowed.
Variant = Literal["honest", "impossible", "tempting"]
VARIANTS: tuple[Variant, ...] = ("honest", "impossible", "tempting")

Source = Literal["humaneval", "mbpp"]


@dataclass(frozen=True)
class TestCase:
    """One test: call the function with these arguments and expect this result."""

    # pytest would otherwise try to collect this class as a test because its name starts with "Test".
    __test__ = False

    args_repr: str  # Python text of the argument list, e.g. "[[1.0, 2.0], 0.5]"
    expected_repr: str  # Python text of the expected return value, e.g. "False"


@dataclass(frozen=True)
class Problem:
    """A coding problem in a common format, whichever dataset it came from."""

    problem_id: str  # e.g. "HumanEval/0" or "Mbpp/2"
    source: Source
    entry_point: str  # name of the function to implement
    starter_code: str  # what the agent starts with: signature and docstring, no body
    canonical_solution: str  # a known-correct full solution, used for sanity checks
    checker_source: str  # imports and the assertion() helper that compares outputs
    atol: float  # absolute tolerance when comparing floating point outputs
    cases: list[TestCase]  # all test cases; the original benchmark tests come first


@dataclass(frozen=True)
class Mutation:
    """Records the one visible test we broke to make an impossible task."""

    case_index: int  # position in the task's visible_cases list
    original_expected_repr: str
    mutated_expected_repr: str


@dataclass(frozen=True)
class Task:
    """One problem in one variant: everything needed to build a workspace and grade it."""

    task_id: str  # e.g. "HumanEval/0::impossible"
    problem_id: str
    source: Source
    variant: Variant
    entry_point: str
    instructions: str  # contents of TASK.md, the note the agent reads
    starter_code: str  # contents of solution.py at the start
    canonical_solution: str
    checker_source: str
    atol: float
    visible_cases: list[TestCase]  # written to test_solution.py, which the agent can see and edit
    hidden_cases: list[TestCase]  # never shown to the agent; they decide real correctness
    mutation: Mutation | None = field(default=None)  # only set for impossible tasks


def task_to_dict(task: Task) -> dict[str, Any]:
    """Turn a Task into plain dicts and lists so it can be written as JSON."""
    return asdict(task)


def task_from_dict(data: dict[str, Any]) -> Task:
    """Rebuild a Task (including its nested TestCase and Mutation objects) from a dict."""
    data = dict(data)  # copy so we do not change the caller's dict
    data["visible_cases"] = [TestCase(**c) for c in data["visible_cases"]]
    data["hidden_cases"] = [TestCase(**c) for c in data["hidden_cases"]]
    if data.get("mutation") is not None:
        data["mutation"] = Mutation(**data["mutation"])
    return Task(**data)


def save_tasks(tasks: Iterable[Task], path: Path) -> int:
    """Write tasks to a JSONL file (one JSON object per line). Returns how many were written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for task in tasks:
            f.write(json.dumps(task_to_dict(task)) + "\n")
            count += 1
    return count


def load_tasks(path: Path) -> list[Task]:
    """Read tasks back from a JSONL file written by save_tasks."""
    with path.open(encoding="utf-8") as f:
        return [task_from_dict(json.loads(line)) for line in f if line.strip()]
