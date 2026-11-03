"""Download HumanEval+ and MBPP+ and convert each problem into our common Problem format.

The datasets come from Hugging Face (evalplus/humanevalplus and evalplus/mbppplus). The
`datasets` library saves them in ~/.cache/huggingface after the first download, so later
runs work offline.

The two datasets look different:
- HumanEval+ gives a `prompt` that is already a function signature plus docstring, and a
  `canonical_solution` that is just the function body.
- MBPP+ gives a one-line English `prompt` and a full solution in `code`. We build the
  starter code ourselves from the solution's function signature.
"""

from __future__ import annotations

import ast
import logging
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from hackguard.tasks.cases import MIN_CASES, UnsupportedTestFormat, parse_evalplus_tests
from hackguard.tasks.schema import Problem

log = logging.getLogger(__name__)

HUMANEVAL_DATASET = "evalplus/humanevalplus"
MBPP_DATASET = "evalplus/mbppplus"

# The line we put where the function body should go in the starter code.
STUB_BODY = "raise NotImplementedError"


@dataclass(frozen=True)
class SkippedProblem:
    """A problem we could not use, and why. Reported so nothing is dropped silently."""

    problem_id: str
    reason: str


def _humaneval_id(row: Mapping[str, Any]) -> str:
    return str(row["task_id"])  # already looks like "HumanEval/0"


def _mbpp_id(row: Mapping[str, Any]) -> str:
    return f"Mbpp/{row['task_id']}"  # MBPP ids are plain numbers, so we add a prefix


def problem_from_humaneval(row: Mapping[str, Any]) -> Problem:
    """Convert one HumanEval+ row. Raises UnsupportedTestFormat if it cannot be used."""
    problem_id = _humaneval_id(row)
    entry_point = str(row["entry_point"])
    prompt = str(row["prompt"])
    parsed = parse_evalplus_tests(str(row["test"]))
    _check_enough_cases(len(parsed.cases))

    # HumanEval prompts end right after the docstring, so we add an indented stub line.
    indent = _body_indent(prompt, entry_point)
    starter_code = prompt.rstrip() + "\n" + indent + STUB_BODY + "\n"

    return Problem(
        problem_id=problem_id,
        source="humaneval",
        entry_point=entry_point,
        starter_code=starter_code,
        canonical_solution=prompt + str(row["canonical_solution"]),
        checker_source=parsed.checker_source,
        atol=parsed.atol,
        cases=parsed.cases,
    )


def problem_from_mbpp(row: Mapping[str, Any]) -> Problem:
    """Convert one MBPP+ row. Raises UnsupportedTestFormat if it cannot be used."""
    problem_id = _mbpp_id(row)
    parsed = parse_evalplus_tests(str(row["test"]), extra_imports=list(row["test_imports"]))
    _check_enough_cases(len(parsed.cases))
    entry_point = parsed.function_name  # MBPP has no entry_point field; the test loop names it
    solution = str(row["code"])

    # Find the solution's definition of the entry function to copy its parameter list.
    func = _find_function(solution, entry_point)
    if func is None:
        raise UnsupportedTestFormat(f"solution does not define {entry_point}")
    params = ast.unparse(func.args)

    # The English prompt goes in the docstring. Escape backslashes and triple quotes so the
    # prompt text cannot break the docstring.
    description = str(row["prompt"]).strip().replace("\\", "\\\\").replace('"""', "'''")
    starter_code = f'def {entry_point}({params}):\n    """{description}\n    """\n    {STUB_BODY}\n'
    ast.parse(starter_code)  # fails loudly if we somehow built invalid Python

    return Problem(
        problem_id=problem_id,
        source="mbpp",
        entry_point=entry_point,
        starter_code=starter_code,
        canonical_solution=solution,
        checker_source=parsed.checker_source,
        atol=parsed.atol,
        cases=parsed.cases,
    )


def load_problems() -> tuple[list[Problem], list[SkippedProblem]]:
    """Download both datasets and convert every problem. Returns (usable, skipped)."""
    # Imported here so the rest of the module (and its tests) work without loading datasets.
    from datasets import load_dataset

    problems: list[Problem] = []
    skipped: list[SkippedProblem] = []
    sources: list[tuple[str, Callable[[Mapping[str, Any]], Problem], Callable[[Mapping[str, Any]], str]]] = [
        (HUMANEVAL_DATASET, problem_from_humaneval, _humaneval_id),
        (MBPP_DATASET, problem_from_mbpp, _mbpp_id),
    ]
    for dataset_name, convert, get_id in sources:
        for row in load_dataset(dataset_name, split="test"):
            try:
                problems.append(convert(row))
            except (UnsupportedTestFormat, SyntaxError) as exc:
                skipped.append(SkippedProblem(get_id(row), str(exc)))
                log.info("Skipping %s: %s", get_id(row), exc)
    return problems, skipped


def _check_enough_cases(n: int) -> None:
    if n < MIN_CASES:
        raise UnsupportedTestFormat(f"only {n} test cases (need {MIN_CASES})")


def _find_function(code: str, name: str) -> ast.FunctionDef | None:
    """Return the top-level function called `name` in some code, or None."""
    for node in ast.parse(code).body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    return None


def _body_indent(prompt: str, entry_point: str) -> str:
    """How far the entry function's body is indented (usually four spaces).

    We read the column of the first statement in the function (the docstring).
    The prompt has no body yet, so we parse it with a temporary `pass` added.
    """
    func = _find_function(prompt.rstrip() + "\n    pass\n", entry_point)
    if func is None or not func.body:
        raise UnsupportedTestFormat(f"prompt does not define {entry_point}")
    return " " * func.body[0].col_offset
