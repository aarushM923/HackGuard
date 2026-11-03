from pathlib import Path

from hackguard.tasks.schema import (
    Mutation,
    Task,
    TestCase,
    load_tasks,
    save_tasks,
    task_from_dict,
    task_to_dict,
)


def _task(mutation: Mutation | None = None) -> Task:
    return Task(
        task_id="HumanEval/0::honest",
        problem_id="HumanEval/0",
        source="humaneval",
        variant="honest",
        entry_point="f",
        instructions="do it",
        starter_code="def f(x):\n    raise NotImplementedError\n",
        canonical_solution="def f(x):\n    return x\n",
        checker_source="def assertion(out, exp, atol):\n    assert out == exp\n",
        atol=0.0,
        visible_cases=[TestCase("[(1, 2)]", "(1, 2)")],
        hidden_cases=[TestCase("[(1, 2)]", "(1, 2)"), TestCase("[{1, 2}]", "{1, 2}")],
        mutation=mutation,
    )


def test_dict_round_trip_keeps_nested_objects() -> None:
    task = _task(Mutation(case_index=0, original_expected_repr="1", mutated_expected_repr="2"))
    assert task_from_dict(task_to_dict(task)) == task


def test_save_and_load_jsonl(tmp_path: Path) -> None:
    tasks = [_task(), _task(Mutation(0, "True", "False"))]
    path = tmp_path / "sub" / "tasks.jsonl"  # parent folder does not exist yet
    assert save_tasks(tasks, path) == 2
    assert load_tasks(path) == tasks


def test_tuples_and_sets_survive_saving(tmp_path: Path) -> None:
    """Values are stored as Python text, so tuples and sets are not turned into lists."""
    path = tmp_path / "tasks.jsonl"
    save_tasks([_task()], path)
    loaded = load_tasks(path)[0]
    assert loaded.hidden_cases[1].expected_repr == "{1, 2}"
    assert loaded.visible_cases[0].expected_repr == "(1, 2)"
