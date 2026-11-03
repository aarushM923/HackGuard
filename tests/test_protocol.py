from hackguard.agent.generation import cut_at_stop
from hackguard.agent.protocol import parse_action, truncate_after_action


def test_parse_write_file_and_content_position() -> None:
    msg = 'I will fix it.\n<write_file path="solution.py">\ndef f():\n    return 1\n</write_file>'
    action = parse_action(msg)
    assert action.kind == "write_file"
    assert action.path == "solution.py"
    assert action.content == "def f():\n    return 1\n"
    assert action.content_start is not None
    assert msg[action.content_start:].startswith("def f():")
    assert msg[action.start:].startswith("<write_file")


def test_code_fence_inside_write_file_is_removed_and_position_still_correct() -> None:
    msg = '<write_file path="solution.py">\n```python\ndef f():\n    return 1\n```\n</write_file>'
    action = parse_action(msg)
    assert action.content == "def f():\n    return 1\n"
    assert action.content_start is not None
    assert msg[action.content_start:].startswith("def f():")


def test_other_actions() -> None:
    assert parse_action("Let me look.\n<read_file> test_solution.py </read_file>").path == "test_solution.py"
    assert parse_action("ok <run_tests/>").kind == "run_tests"
    assert parse_action("done <submit />").kind == "submit"
    assert parse_action("```python\nprint(1)\n```").kind == "invalid"


def test_first_action_wins_and_rest_is_dropped() -> None:
    msg = "<run_tests/> and then <submit/>"
    action = parse_action(msg)
    assert action.kind == "run_tests"
    assert truncate_after_action(msg, action) == "<run_tests/>"


def test_cut_at_stop_keeps_the_stop_string() -> None:
    assert cut_at_stop("a <run_tests/> extra text") == "a <run_tests/>"
    assert cut_at_stop("no action here") == "no action here"
