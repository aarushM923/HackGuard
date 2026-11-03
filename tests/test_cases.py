import pytest

from conftest import ADD_INPUTS, ADD_RESULTS, humaneval_test_code, mbpp_test_code
from hackguard.tasks.cases import (
    MAX_READABLE_REPR,
    NUM_TEMPTING_LONG,
    NUM_TEMPTING_SHORT,
    NUM_VISIBLE,
    UnsupportedTestFormat,
    hidden_cases,
    parse_evalplus_tests,
    tempting_cases,
    to_repr,
    visible_cases,
)
from hackguard.tasks.schema import TestCase


def test_parse_humaneval_layout() -> None:
    parsed = parse_evalplus_tests(humaneval_test_code(ADD_INPUTS, ADD_RESULTS))
    assert parsed.function_name == "candidate"
    assert parsed.atol == 0.0
    assert len(parsed.cases) == len(ADD_INPUTS)
    assert parsed.cases[0] == TestCase("[1, 2]", "3")
    # The checker keeps the helper and imports but drops check() and the data.
    assert "def assertion(" in parsed.checker_source
    assert "import numpy as np" in parsed.checker_source
    assert "def check(" not in parsed.checker_source
    assert "inputs" not in parsed.checker_source


def test_parse_mbpp_layout_with_extra_imports_and_tolerance() -> None:
    code = mbpp_test_code("area", [[1.0], [2.0]], [3.14, 12.56], atol=1e-4)
    parsed = parse_evalplus_tests(code, extra_imports=["import math"])
    assert parsed.function_name == "area"
    assert parsed.atol == 1e-4
    assert parsed.checker_source.startswith("import math")
    assert "results" not in parsed.checker_source
    assert "for i" not in parsed.checker_source


def test_computed_results_are_unsupported() -> None:
    """Some problems compute expected results with a reference function instead of listing them."""
    code = humaneval_test_code([[1, 2]], [3]).replace("results = [3]", "results = [ref(*x) for x in inputs]")
    with pytest.raises(UnsupportedTestFormat, match="not a plain data list"):
        parse_evalplus_tests(code)


def test_custom_loop_is_unsupported() -> None:
    code = humaneval_test_code([[1, 2]], [3]).replace(
        "assertion(candidate(*inp), exp, 0)", "assert abs(candidate(*inp) - exp) < 1"
    )
    with pytest.raises(UnsupportedTestFormat, match="custom shape"):
        parse_evalplus_tests(code)


def test_mismatched_list_lengths_are_unsupported() -> None:
    with pytest.raises(UnsupportedTestFormat, match="inputs but"):
        parse_evalplus_tests(humaneval_test_code([[1, 2], [3, 4]], [3]))


def test_missing_assertion_helper_is_unsupported() -> None:
    code = humaneval_test_code([[1, 2]], [3]).replace("def assertion(", "def other_name(")
    with pytest.raises(UnsupportedTestFormat):
        parse_evalplus_tests(code)


def test_to_repr_accepts_plain_values_and_rejects_infinity() -> None:
    assert to_repr([(1, "a"), {"k": None}]) == "[(1, 'a'), {'k': None}]"
    with pytest.raises(UnsupportedTestFormat):
        to_repr(float("inf"))  # prints as "inf", which cannot be read back as data


def _cases(arg_reprs: list[str]) -> list[TestCase]:
    return [TestCase(a, "0") for a in arg_reprs]


def test_visible_and_hidden_split() -> None:
    cases = _cases([f"[{i}]" for i in range(20)])
    assert visible_cases(cases) == cases[:NUM_VISIBLE]
    assert hidden_cases(cases) == cases  # hidden is the full suite


def test_tempting_adds_shortest_and_longest_readable_cases() -> None:
    base = ["[1]", "[2]", "[3]"]
    others = ["[]", "[[1, 2, 3]]", "[0]", "['ab']", "['abcdefgh']", "['x' * 1]", "['" + "y" * 400 + "']"]
    chosen = [c.args_repr for c in tempting_cases(_cases(base + others))]

    assert chosen[:NUM_VISIBLE] == base
    assert len(chosen) == NUM_VISIBLE + NUM_TEMPTING_SHORT + NUM_TEMPTING_LONG
    assert chosen[NUM_VISIBLE] == "[]"  # the very shortest input comes first
    assert "['abcdefgh']" in chosen and "[[1, 2, 3]]" in chosen  # longest readable ones
    assert all(len(a) <= MAX_READABLE_REPR for a in chosen)  # the 400-char one is left out
    assert len(set(chosen)) == len(chosen)  # no duplicates


def test_tempting_skips_duplicates_of_visible_cases() -> None:
    cases = _cases(["[1]", "[2]", "[3]", "[1]", "[2]", "[4]"])
    assert [c.args_repr for c in tempting_cases(cases)] == ["[1]", "[2]", "[3]", "[4]"]
