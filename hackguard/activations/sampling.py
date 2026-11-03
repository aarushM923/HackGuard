"""Choose which token positions to save activations for (spec Step 3.3).

Positives: in each hack trajectory, the last `window` tokens the agent wrote *before* the
first hacking token. These are moments where the model is about to cheat but has not
written any cheating text yet, so a probe that fires here is predicting, not detecting.

Negatives: in each honest trajectory, `window` agent-written tokens ending at a "matched"
position. We pick how many agent tokens into the episode that position is by copying the
hack position of a random hack trajectory (of the same task variant when possible). This
way positives and negatives come from similar points in an episode, so the probe cannot
just learn "late in the episode means hacking".

We only use tokens the agent wrote itself (assistant messages): those are the model's own
decisions. Saving every token of every layer would take hundreds of gigabytes; windows
keep it to a few gigabytes.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from hackguard.activations.tokens import TokenizedTrajectory
from hackguard.trajectories.schema import Label, Trajectory

USABLE_LABELS = frozenset({"hack", "honest_success", "honest_fail"})


@dataclass(frozen=True)
class Sample:
    trajectory_id: str
    problem_id: str
    variant: str
    prompt_name: str
    label: int  # 1 = this trajectory hacks, 0 = it does not
    hack_types: str  # space-separated, empty for honest trajectories
    token_index: int  # position in this model's token sequence
    message_index: int  # tokenizer-independent position: which message ...
    char_end: int  # ... and the character where this token ends inside it
    tokens_before: int  # agent tokens between this token and the hack (or matched point)


def hack_rank(traj: Trajectory, label: Label, tt: TokenizedTrajectory, assistant: list[int]) -> int | None:
    """How many agent-written tokens come before the first hacking token."""
    if label.first_hack is None:
        return None
    t_hack = tt.message_char_to_token(label.first_hack.message_index, label.first_hack.char_offset)
    return sum(1 for t in assistant if t < t_hack)


def sample_positions(
    traj: Trajectory,
    label: Label,
    tt: TokenizedTrajectory,
    window: int,
    reference_ranks: dict[str, list[int]],
    rng: random.Random,
) -> list[Sample]:
    """Pick the window of positions to save for one trajectory (empty if unusable)."""
    if label.label not in USABLE_LABELS:
        return []
    assistant = tt.assistant_tokens()
    if not assistant:
        return []

    if label.label == "hack":
        end = hack_rank(traj, label, tt, assistant)
        if not end:  # None, or the hack is the very first token (nothing before it)
            return []
    else:
        refs = reference_ranks.get(traj.variant) or reference_ranks.get("all") or [len(assistant)]
        end = min(rng.choice(refs), len(assistant))

    samples = []
    for rank in range(max(0, end - window), end):
        token = assistant[rank]
        message_index, char_end = tt.token_to_message_char(token)
        samples.append(Sample(
            trajectory_id=traj.trajectory_id,
            problem_id=traj.problem_id,
            variant=traj.variant,
            prompt_name=traj.prompt_name,
            label=int(label.label == "hack"),
            hack_types=" ".join(label.hack_types),
            token_index=token,
            message_index=message_index,
            char_end=char_end,
            tokens_before=end - rank,
        ))
    return samples
