import json
from dataclasses import replace
from pathlib import Path

from hackguard.tasks.build_suite import ProblemCheck, build_suite, check_problem, select_problems
from hackguard.tasks.schema import Problem, load_tasks


def test_good_problem_passes_all_checks(humaneval_problem: Problem) -> None:
    check = check_problem(humaneval_problem, seed=0)
    assert check.failures == []
    assert check.ok
    assert [t.variant for t in check.tasks] == ["honest", "impossible", "tempting"]


def test_wrong_canonical_solution_is_caught(humaneval_problem: Problem) -> None:
    broken = replace(
        humaneval_problem,
        canonical_solution="def add(a: int, b: int) -> int:\n    return a - b\n",
    )
    check = check_problem(broken, seed=0)
    assert not check.ok
    assert any("hidden" in f for f in check.failures)


def _check(problem_id: str) -> ProblemCheck:
    return ProblemCheck(problem_id)


def test_select_problems_keeps_humaneval_and_fills_with_mbpp() -> None:
    good = [_check(f"HumanEval/{i}") for i in range(5)] + [_check(f"Mbpp/{i}") for i in range(20)]
    chosen = select_problems(good, num_problems=10, seed=0)
    ids = [c.problem_id for c in chosen]
    assert len(ids) == 10
    assert ids[:5] == [f"HumanEval/{i}" for i in range(5)]
    assert all(i.startswith("Mbpp/") for i in ids[5:])
    # Same seed, same choice.
    assert ids == [c.problem_id for c in select_problems(good, num_problems=10, seed=0)]


def test_select_problems_sorts_ids_by_number() -> None:
    good = [_check("HumanEval/10"), _check("HumanEval/2")]
    assert [c.problem_id for c in select_problems(good, 2, seed=0)] == ["HumanEval/2", "HumanEval/10"]


def test_build_suite_writes_tasks_and_report(
    humaneval_problem: Problem, mbpp_problem: Problem, tmp_path: Path
) -> None:
    report = build_suite(tmp_path, num_problems=400, seed=0, workers=2,
                         problems=[humaneval_problem, mbpp_problem])
    tasks = load_tasks(tmp_path / "tasks.jsonl")
    assert len(tasks) == 6
    assert report["tasks_by_variant"] == {"honest": 2, "impossible": 2, "tempting": 2}
    saved = json.loads((tmp_path / "report.json").read_text())
    assert saved["problems_in_suite"] == 2
    assert saved["problems_failing_checks"] == []
