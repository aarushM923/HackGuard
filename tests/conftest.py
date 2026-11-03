"""Shared test fixtures: small fake dataset rows in the same shape as HumanEval+ and MBPP+.

pytest automatically loads this file, and any test can ask for a fixture by naming it as a
function argument (e.g. `def test_x(humaneval_row): ...`).
"""

from __future__ import annotations

from typing import Any

import pytest

from hackguard.tasks.loaders import problem_from_humaneval, problem_from_mbpp
from hackguard.tasks.schema import Problem

# The comparison helper, shortened from the real EvalPlus one.
ASSERTION_HELPER = '''
import numpy as np

def assertion(out, exp, atol):
    if atol != 0 and isinstance(exp, float):
        assert np.allclose(out, exp, rtol=1e-07, atol=atol)
    else:
        assert out == exp
'''

# Twelve cases for add(a, b). The first three play the role of the original benchmark tests.
ADD_INPUTS = [[1, 2], [0, 0], [-1, 1], [5, 5], [100, 200], [-50, -50], [7, 8], [3, 4],
              [10, -3], [999, 1], [2, 2], [123456, 654321]]
ADD_RESULTS = [a + b for a, b in ADD_INPUTS]


def humaneval_test_code(inputs: list[Any], results: list[Any], atol: float = 0) -> str:
    """Test code laid out like HumanEval+: lists and loop inside def check(candidate)."""
    return f"""{ASSERTION_HELPER}

def check(candidate):
    inputs = {inputs!r}
    results = {results!r}
    for i, (inp, exp) in enumerate(zip(inputs, results)):
        assertion(candidate(*inp), exp, {atol!r})
"""


def mbpp_test_code(func: str, inputs: list[Any], results: list[Any], atol: float = 0) -> str:
    """Test code laid out like MBPP+: lists and loop at the top level."""
    return f"""{ASSERTION_HELPER}

inputs = {inputs!r}
results = {results!r}
for i, (inp, exp) in enumerate(zip(inputs, results)):
    assertion({func}(*inp), exp, {atol!r})
"""


@pytest.fixture
def humaneval_row() -> dict[str, Any]:
    return {
        "task_id": "HumanEval/900",
        "entry_point": "add",
        "prompt": 'def add(a: int, b: int) -> int:\n    """Add two numbers.\n    >>> add(1, 2)\n    3\n    """\n',
        "canonical_solution": "    return a + b\n",
        "test": humaneval_test_code(ADD_INPUTS, ADD_RESULTS),
    }


@pytest.fixture
def mbpp_row() -> dict[str, Any]:
    inputs = [[i, i + 1] for i in range(12)]
    return {
        "task_id": 901,
        "prompt": "Write a function to multiply two numbers.",
        "code": "import math\n\ndef mul(x, y):\n  return x * y\n",
        "test_imports": ["import math"],
        "test": mbpp_test_code("mul", inputs, [a * b for a, b in inputs]),
    }


@pytest.fixture
def humaneval_problem(humaneval_row: dict[str, Any]) -> Problem:
    return problem_from_humaneval(humaneval_row)


@pytest.fixture
def mbpp_problem(mbpp_row: dict[str, Any]) -> Problem:
    return problem_from_mbpp(mbpp_row)
