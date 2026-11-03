import ast
from typing import Any

import pytest

from conftest import mbpp_test_code
from hackguard.tasks.cases import UnsupportedTestFormat
from hackguard.tasks.loaders import STUB_BODY, load_problems, problem_from_humaneval, problem_from_mbpp


def test_humaneval_problem(humaneval_row: dict[str, Any]) -> None:
    p = problem_from_humaneval(humaneval_row)
    assert p.problem_id == "HumanEval/900"
    assert p.source == "humaneval"
    assert p.entry_point == "add"
    assert p.starter_code.endswith(f"    {STUB_BODY}\n")
    assert '"""Add two numbers.' in p.starter_code  # docstring kept
    ast.parse(p.starter_code)
    assert p.canonical_solution == humaneval_row["prompt"] + "    return a + b\n"
    assert len(p.cases) == 12


def test_mbpp_problem(mbpp_row: dict[str, Any]) -> None:
    p = problem_from_mbpp(mbpp_row)
    assert p.problem_id == "Mbpp/901"
    assert p.entry_point == "mul"  # read from the test loop
    assert p.starter_code.startswith("def mul(x, y):\n")
    assert "Write a function to multiply two numbers." in p.starter_code
    assert "import math" in p.checker_source  # from test_imports
    assert p.canonical_solution == mbpp_row["code"]


def test_mbpp_prompt_with_quotes_and_backslashes_stays_valid(mbpp_row: dict[str, Any]) -> None:
    mbpp_row["prompt"] = 'Handle """quotes""" and a backslash \\n here.'
    ast.parse(problem_from_mbpp(mbpp_row).starter_code)


def test_mbpp_solution_without_entry_function_is_unsupported(mbpp_row: dict[str, Any]) -> None:
    mbpp_row["code"] = "def other(x, y):\n  return x * y\n"
    with pytest.raises(UnsupportedTestFormat, match="does not define mul"):
        problem_from_mbpp(mbpp_row)


def test_too_few_cases_is_unsupported(mbpp_row: dict[str, Any]) -> None:
    mbpp_row["test"] = mbpp_test_code("mul", [[1, 2]], [2])
    with pytest.raises(UnsupportedTestFormat, match="only 1 test cases"):
        problem_from_mbpp(mbpp_row)


# A few dataset solutions write regex strings like '\w' without an r prefix. Python warns
# about that style when parsing them; it is the dataset's code and harmless.
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_real_datasets_load() -> None:
    """Loads the real HumanEval+ and MBPP+ (downloads once, then uses the local cache)."""
    problems, skipped = load_problems()
    ids = {p.problem_id for p in problems}
    assert len(problems) >= 500
    assert len(problems) + len(skipped) == 164 + 378  # every row is either used or explained
    assert "HumanEval/0" in ids and "Mbpp/2" in ids
    for p in problems:
        ast.parse(p.starter_code)
        ast.parse(p.canonical_solution)
