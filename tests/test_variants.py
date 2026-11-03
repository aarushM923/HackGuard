import ast
import random
from dataclasses import replace
from typing import Any

import pytest

from hackguard.tasks.schema import Problem, TestCase
from hackguard.tasks.variants import (
    TEMPTING_NOTE,
    CannotMutate,
    make_all_variants,
    make_honest,
    make_impossible,
    make_tempting,
    mutate_value,
)


@pytest.mark.parametrize(
    "value, expected",
    [
        (True, False),
        (False, True),
        (5, 6),
        (2.5, 5.0),
        (1e10, 2e10),  # big floats move by their own size, so the tolerance cannot hide it
        (0.0, 1.0),
        ("abc", "abcx"),
        ("", "x"),
        ([1, 2], [2, 2]),
        ((1, 2), (2, 2)),
        ([[True]], [[False]]),
        ({"a": 1}, {"a": 2}),
    ],
)
def test_mutate_value(value: Any, expected: Any) -> None:
    result = mutate_value(value)
    assert result == expected
    assert type(result) is type(value)


@pytest.mark.parametrize("value", [None, [], (), {}, {1, 2}, [None]])
def test_mutate_value_refuses_unsafe_values(value: Any) -> None:
    assert mutate_value(value) is None


def test_honest_task(humaneval_problem: Problem) -> None:
    task = make_honest(humaneval_problem)
    assert task.task_id == "HumanEval/900::honest"
    assert task.visible_cases == humaneval_problem.cases[:3]
    assert task.hidden_cases == humaneval_problem.cases
    assert task.mutation is None
    assert TEMPTING_NOTE not in task.instructions
    assert "`add`" in task.instructions


def test_impossible_task_breaks_exactly_one_visible_case(humaneval_problem: Problem) -> None:
    honest = make_honest(humaneval_problem)
    task = make_impossible(humaneval_problem, random.Random(0))
    assert task.mutation is not None
    changed = [i for i, (a, b) in enumerate(zip(honest.visible_cases, task.visible_cases)) if a != b]
    assert changed == [task.mutation.case_index]
    broken = task.visible_cases[task.mutation.case_index]
    assert broken.expected_repr == task.mutation.mutated_expected_repr
    assert ast.literal_eval(task.mutation.original_expected_repr) + 1 == ast.literal_eval(broken.expected_repr)
    # Hidden tests keep the correct values.
    assert task.hidden_cases == humaneval_problem.cases


def test_impossible_is_reproducible_with_same_seed(humaneval_problem: Problem) -> None:
    a = make_impossible(humaneval_problem, random.Random(42))
    b = make_impossible(humaneval_problem, random.Random(42))
    assert a == b


def test_impossible_skips_cases_it_cannot_mutate(humaneval_problem: Problem) -> None:
    cases = [TestCase("[1]", "None"), TestCase("[2]", "[]"), TestCase("[3]", "7")] + humaneval_problem.cases[3:]
    task = make_impossible(replace(humaneval_problem, cases=cases), random.Random(0))
    assert task.mutation is not None
    assert task.mutation.case_index == 2  # the only mutable one
    assert task.mutation.mutated_expected_repr == "8"


def test_impossible_raises_when_nothing_can_be_mutated(humaneval_problem: Problem) -> None:
    cases = [TestCase(f"[{i}]", "None") for i in range(12)]
    with pytest.raises(CannotMutate):
        make_impossible(replace(humaneval_problem, cases=cases), random.Random(0))


def test_tempting_task(humaneval_problem: Problem) -> None:
    task = make_tempting(humaneval_problem)
    assert task.instructions.endswith(TEMPTING_NOTE)
    assert len(task.visible_cases) > 3  # extra edge cases added
    assert task.visible_cases[:3] == humaneval_problem.cases[:3]


def test_all_variants_share_hidden_tests(mbpp_problem: Problem) -> None:
    tasks = make_all_variants(mbpp_problem, random.Random(0))
    assert [t.variant for t in tasks] == ["honest", "impossible", "tempting"]
    assert all(t.hidden_cases == mbpp_problem.cases for t in tasks)
