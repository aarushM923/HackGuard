import random
from pathlib import Path
from typing import Any

from conftest import ScriptedGenerator
from hackguard.agent.loop import EpisodeSpec, run_episodes, safe_path, turn_seed
from hackguard.tasks.schema import Problem
from hackguard.tasks.variants import make_honest, make_impossible

CORRECT = '<write_file path="solution.py">\ndef add(a, b):\n    return a + b\n</write_file>'


def _run(task: Any, tokenizer: Any, replies: list[str], max_turns: int = 5, max_context: int = 8192) -> Any:
    gen = ScriptedGenerator(tokenizer, lambda prompt, turn: replies[min(turn, len(replies) - 1)])
    [traj] = run_episodes([EpisodeSpec(task, "neutral", 0)], gen, "fake-model", max_turns, 256, max_context)
    return traj


def test_honest_episode(humaneval_problem: Problem, qwen_tokenizer: Any) -> None:
    traj = _run(make_honest(humaneval_problem), qwen_tokenizer, [f"Writing it.\n{CORRECT}", "<run_tests/>", "<submit/>"])
    assert [a.kind for a in traj.actions] == ["write_file", "run_tests", "submit"]
    assert traj.end_reason == "submit"
    assert "3 passed" in traj.actions[1].observation
    assert traj.hidden.all_passed and traj.visible_original.all_passed and traj.visible_workspace.all_passed
    assert "return a + b" in traj.final_files["solution.py"]
    # Roles alternate system, user, then assistant/user pairs, ending with the submit.
    assert [m["role"] for m in traj.messages] == ["system", "user", "assistant", "user", "assistant", "user", "assistant"]
    action = traj.actions[0]
    assert traj.messages[action.message_index]["content"][action.char_start:].startswith("<write_file")


def test_actions_record_positions_of_written_content(humaneval_problem: Problem, qwen_tokenizer: Any) -> None:
    traj = _run(make_honest(humaneval_problem), qwen_tokenizer, [f"Plan first.\n{CORRECT}", "<submit/>"])
    a = traj.actions[0]
    assert traj.messages[a.message_index]["content"][a.content_start:].startswith("def add(a, b):")


def test_text_after_the_action_is_dropped(humaneval_problem: Problem, qwen_tokenizer: Any) -> None:
    traj = _run(make_honest(humaneval_problem), qwen_tokenizer, ["<run_tests/> and the tests said ... <submit/>", "<submit/>"])
    assert traj.messages[2]["content"] == "<run_tests/>"


def test_impossible_task_cannot_pass_honestly(humaneval_problem: Problem, qwen_tokenizer: Any) -> None:
    task = make_impossible(humaneval_problem, random.Random(0))
    traj = _run(task, qwen_tokenizer, [CORRECT, "<run_tests/>", "<submit/>"])
    assert traj.hidden.all_passed
    assert not traj.visible_original.all_passed
    assert "1 failed" in traj.actions[1].observation


def test_invalid_replies_end_the_episode(humaneval_problem: Problem, qwen_tokenizer: Any) -> None:
    traj = _run(make_honest(humaneval_problem), qwen_tokenizer, ["```python\nprint(1)\n```"])
    assert traj.end_reason == "invalid_actions"
    assert [a.kind for a in traj.actions] == ["invalid", "invalid"]
    assert "No valid action" in traj.actions[0].observation


def test_max_turns(humaneval_problem: Problem, qwen_tokenizer: Any) -> None:
    traj = _run(make_honest(humaneval_problem), qwen_tokenizer, ["<run_tests/>"], max_turns=3)
    assert traj.end_reason == "max_turns" and len(traj.actions) == 3


def test_context_limit(humaneval_problem: Problem, qwen_tokenizer: Any) -> None:
    traj = _run(make_honest(humaneval_problem), qwen_tokenizer, ["<run_tests/>"], max_context=300)
    assert traj.end_reason == "context_limit" and traj.actions == []


def test_agent_cannot_escape_the_folder(humaneval_problem: Problem, qwen_tokenizer: Any, tmp_path: Path) -> None:
    assert safe_path(tmp_path, "../outside.txt") is None
    assert safe_path(tmp_path, "/etc/passwd") is None
    assert safe_path(tmp_path, "sub/ok.py") == (tmp_path / "sub/ok.py").resolve()
    traj = _run(make_honest(humaneval_problem), qwen_tokenizer,
                ['<write_file path="../escape.py">\nx = 1\n</write_file>', "<submit/>"])
    assert "only access files inside" in traj.actions[0].observation


def test_turn_seeds_differ_and_repeat() -> None:
    assert turn_seed("a", 0) == turn_seed("a", 0)
    assert turn_seed("a", 0) != turn_seed("a", 1) != turn_seed("b", 0)
