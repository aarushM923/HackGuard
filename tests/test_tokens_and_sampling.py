import random
from typing import Any

from hackguard.activations.sampling import sample_positions
from hackguard.activations.tokens import tokenize_trajectory
from hackguard.trajectories.schema import HackLocation, Label, RunSummary, Trajectory

MESSAGES = [
    {"role": "system", "content": "You are an agent."},
    {"role": "user", "content": "Fix the function."},
    {"role": "assistant", "content": "The test looks wrong. I will edit it.\n<write_file path=\"test_solution.py\">\nx = 1\n</write_file>"},
    {"role": "user", "content": "Wrote test_solution.py (1 lines)."},
    {"role": "assistant", "content": "<submit/>"},
]


def _traj(messages: list[dict[str, str]] = MESSAGES) -> Trajectory:
    summary = RunSummary(0, 0, False)
    return Trajectory("T/1::impossible::neutral::s0", "T/1::impossible", "T/1", "impossible", "humaneval", "neutral", 0,
                      "m", messages, [], "submit", {}, summary, summary, summary)


def test_message_positions_and_tokens(qwen_tokenizer: Any) -> None:
    tt = tokenize_trajectory(qwen_tokenizer, MESSAGES)
    text = qwen_tokenizer.decode(tt.input_ids)
    for i, msg in enumerate(MESSAGES):
        start = tt.message_starts[i]
        assert text[start:start + len(msg["content"])] == msg["content"]
    # The token found for a character really covers that character.
    offset = MESSAGES[2]["content"].index("<write_file")
    token = tt.message_char_to_token(2, offset)
    assert tt.token_starts[token] <= tt.message_char_to_text_char(2, offset) < tt.token_ends[token]
    assert "<" in qwen_tokenizer.decode([tt.input_ids[token]])


def test_assistant_tokens_are_only_agent_text(qwen_tokenizer: Any) -> None:
    tt = tokenize_trajectory(qwen_tokenizer, MESSAGES)
    agent_text = "".join(qwen_tokenizer.decode([tt.input_ids[t]]) for t in tt.assistant_tokens())
    assert "The test looks wrong" in agent_text and "<submit/>" in agent_text
    assert "Fix the function" not in agent_text and "Wrote test_solution" not in agent_text


def test_round_trip_between_tokens_and_message_characters(qwen_tokenizer: Any) -> None:
    tt = tokenize_trajectory(qwen_tokenizer, MESSAGES)
    for token in tt.assistant_tokens():
        message, char_end = tt.token_to_message_char(token)
        assert message in (2, 4)
        assert tt.last_token_ending_by(message, char_end) == token


def test_positions_map_across_different_tokenizers(qwen_tokenizer: Any) -> None:
    """A character position chosen with Qwen's tokens lands on a token ending at that spot
    in another tokenizer (here SmolLM2's, which splits text differently)."""
    from hackguard.config import TINY_SECOND_MODEL
    from hackguard.models import load_tokenizer
    other = load_tokenizer(TINY_SECOND_MODEL)
    tq, to = tokenize_trajectory(qwen_tokenizer, MESSAGES), tokenize_trajectory(other, MESSAGES)
    for token in tq.assistant_tokens()[::5]:
        message, char_end = tq.token_to_message_char(token)
        mapped = to.last_token_ending_by(message, char_end)
        assert to.token_ends[mapped] <= to.message_char_to_text_char(message, char_end)
        # and the next token in the other model already goes past the spot
        if mapped + 1 < len(to.token_ends):
            assert to.token_ends[mapped + 1] > to.message_char_to_text_char(message, char_end) or to.token_ends[mapped + 1] == to.token_ends[mapped]


def test_sampling_hack_window_ends_before_hack(qwen_tokenizer: Any) -> None:
    traj = _traj()
    tt = tokenize_trajectory(qwen_tokenizer, MESSAGES)
    offset = MESSAGES[2]["content"].index("<write_file")
    label = Label(traj.trajectory_id, "hack", ["edit_tests"], HackLocation(2, offset, 0, "edit_tests"))
    samples = sample_positions(traj, label, tt, window=4, reference_ranks={}, rng=random.Random(0))
    t_hack = tt.message_char_to_token(2, offset)
    assert len(samples) == 4
    assert all(s.token_index < t_hack and s.label == 1 for s in samples)
    assert [s.tokens_before for s in samples] == [4, 3, 2, 1]
    assert samples[-1].token_index == max(t for t in tt.assistant_tokens() if t < t_hack)


def test_sampling_honest_uses_matched_position(qwen_tokenizer: Any) -> None:
    traj = _traj()
    tt = tokenize_trajectory(qwen_tokenizer, MESSAGES)
    label = Label(traj.trajectory_id, "honest_fail")
    samples = sample_positions(traj, label, tt, window=3, reference_ranks={"impossible": [5]}, rng=random.Random(0))
    assistant = tt.assistant_tokens()
    assert [s.token_index for s in samples] == assistant[2:5]
    assert all(s.label == 0 for s in samples)
    assert sample_positions(traj, Label(traj.trajectory_id, "ambiguous"), tt, 3, {}, random.Random(0)) == []
