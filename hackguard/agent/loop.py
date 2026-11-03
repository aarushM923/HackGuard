"""The agent loop: let the model act in a task folder, turn by turn, and record everything.

One turn of one episode:
  1. render the conversation with the chat template and ask the model for a reply,
  2. find the first action in the reply (and drop anything written after it),
  3. carry out the action in the task folder (read, write, run tests or submit),
  4. add the action's result to the conversation as the next user message.

Many episodes run together: in each turn we send every unfinished episode's prompt to the
generator as one batch, which is far faster on a GPU than one prompt at a time. Running
tests (the slow part on the CPU side) happens in parallel threads.
"""

from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from hackguard.agent.generation import Generator
from hackguard.agent.prompts import first_user_message, system_prompt
from hackguard.agent.protocol import Action, parse_action, truncate_after_action
from hackguard.models import chat_text
from hackguard.tasks.schema import Task
from hackguard.tasks.workspace import (
    LOCAL_RUNNER,
    SOLUTION_FILE,
    VISIBLE_TEST_FILE,
    Runner,
    TestRunResult,
    create_workspace,
    grade,
    remove_workspace,
    run_cases,
    run_tests,
)
from hackguard.trajectories.schema import ActionRecord, RunSummary, Trajectory

MAX_FILE_CHARS_SHOWN = 6000
MAX_INVALID_IN_A_ROW = 2
TEST_TIMEOUT_SECONDS = 60.0

INVALID_ACTION_HELP = (
    "No valid action found. End your reply with exactly one of: "
    "<read_file>FILE</read_file>, <write_file path=\"FILE\">...</write_file>, <run_tests/>, <submit/>"
)


@dataclass(frozen=True)
class EpisodeSpec:
    task: Task
    prompt_name: str
    seed: int

    @property
    def trajectory_id(self) -> str:
        return f"{self.task.task_id}::{self.prompt_name}::s{self.seed}"


@dataclass
class _Episode:
    """Everything that changes while one episode runs."""

    spec: EpisodeSpec
    folder: Path
    messages: list[dict[str, str]]
    actions: list[ActionRecord] = field(default_factory=list)
    done: bool = False
    end_reason: str = "max_turns"
    invalid_in_a_row: int = 0


def turn_seed(trajectory_id: str, turn: int) -> int:
    """A different but reproducible random seed for every episode and turn."""
    digest = hashlib.sha256(f"{trajectory_id}|{turn}".encode()).hexdigest()
    return int(digest[:8], 16)  # first 32 bits of the hash as a number


def safe_path(folder: Path, path: str) -> Path | None:
    """Resolve a path the agent gave, or None if it points outside the task folder.

    This blocks paths like "../../etc/passwd" or "/home/user/.ssh/id_rsa".
    """
    if not path or Path(path).is_absolute():
        return None
    target = (folder / path).resolve()
    return target if target.is_relative_to(folder.resolve()) else None


def execute(action: Action, folder: Path, runner: Runner) -> str:
    """Carry out one action in the task folder and return what the agent sees next."""
    if action.kind == "read_file":
        target = safe_path(folder, action.path or "")
        if target is None:
            return "Error: you can only access files inside the task folder."
        if not target.is_file():
            return f"Error: {action.path} does not exist."
        text = target.read_text(encoding="utf-8", errors="replace")
        return text[:MAX_FILE_CHARS_SHOWN] + ("\n[...file truncated...]" if len(text) > MAX_FILE_CHARS_SHOWN else "")
    if action.kind == "write_file":
        target = safe_path(folder, action.path or "")
        if target is None:
            return "Error: you can only access files inside the task folder."
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(action.content or "", encoding="utf-8")
        return f"Wrote {action.path} ({len((action.content or '').splitlines())} lines)."
    if action.kind == "run_tests":
        result = run_tests(folder, VISIBLE_TEST_FILE, timeout=TEST_TIMEOUT_SECONDS, runner=runner, show_failures=True)
        return result.output.strip() or "No test output."
    if action.kind == "submit":
        return ""
    return INVALID_ACTION_HELP


def run_episodes(
    specs: list[EpisodeSpec],
    generator: Generator,
    model_id: str,
    max_turns: int,
    max_new_tokens: int,
    max_context_tokens: int,
    runner: Runner = LOCAL_RUNNER,
    workers: int = 8,
    workdir: Path | None = None,
) -> list[Trajectory]:
    """Run a batch of episodes to completion and return their trajectories."""
    episodes = [
        _Episode(
            spec=spec,
            folder=create_workspace(spec.task, parent=workdir),
            messages=[
                {"role": "system", "content": system_prompt(spec.prompt_name, max_turns)},
                {"role": "user", "content": first_user_message(spec.task)},
            ],
        )
        for spec in specs
    ]
    tokenizer = generator.tokenizer
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for turn in range(max_turns):
            active = [e for e in episodes if not e.done]
            if not active:
                break

            # Build prompts; stop any episode whose conversation no longer fits the context window.
            ready: list[tuple[_Episode, str]] = []
            for ep in active:
                prompt = chat_text(tokenizer, ep.messages, add_generation_prompt=True)
                if len(tokenizer(prompt, add_special_tokens=False)["input_ids"]) + max_new_tokens > max_context_tokens:
                    ep.done, ep.end_reason = True, "context_limit"
                else:
                    ready.append((ep, prompt))
            if not ready:
                break

            replies = generator.generate(
                [prompt for _, prompt in ready],
                [turn_seed(ep.spec.trajectory_id, turn) for ep, _ in ready],
            )

            parsed = []
            for (ep, _), reply in zip(ready, replies):
                action = parse_action(reply)
                ep.messages.append({"role": "assistant", "content": truncate_after_action(reply, action)})
                parsed.append((ep, action))

            # Carry out all actions in parallel (each touches only its own folder).
            observations = list(pool.map(lambda pair: execute(pair[1], pair[0].folder, runner), parsed))

            for (ep, action), observation in zip(parsed, observations):
                ep.actions.append(ActionRecord(
                    turn=turn, kind=action.kind, message_index=len(ep.messages) - 1,
                    char_start=action.start, char_end=action.end, path=action.path,
                    content=action.content, content_start=action.content_start, observation=observation,
                ))
                if action.kind == "submit":
                    ep.done, ep.end_reason = True, "submit"
                    continue
                ep.invalid_in_a_row = ep.invalid_in_a_row + 1 if action.kind == "invalid" else 0
                if ep.invalid_in_a_row >= MAX_INVALID_IN_A_ROW:
                    ep.done, ep.end_reason = True, "invalid_actions"
                    continue
                ep.messages.append({"role": "user", "content": observation})

        trajectories = list(pool.map(lambda ep: _finish(ep, model_id, runner), episodes))
    return trajectories


def _finish(ep: _Episode, model_id: str, runner: Runner) -> Trajectory:
    """Run the three final test checks, snapshot the files, and delete the folder."""
    task = ep.spec.task
    try:
        solution_path = ep.folder / SOLUTION_FILE
        solution = solution_path.read_text(encoding="utf-8", errors="replace") if solution_path.is_file() else ""
        workspace_run = run_tests(ep.folder, VISIBLE_TEST_FILE, timeout=TEST_TIMEOUT_SECONDS, runner=runner)
        original_run = run_cases(task, solution, task.visible_cases, timeout=TEST_TIMEOUT_SECONDS, runner=runner)
        hidden_run = grade(task, solution, runner=runner)
        final_files = _snapshot(ep.folder)
    finally:
        remove_workspace(ep.folder)
    return Trajectory(
        trajectory_id=ep.spec.trajectory_id,
        task_id=task.task_id,
        problem_id=task.problem_id,
        variant=task.variant,
        source=task.source,
        prompt_name=ep.spec.prompt_name,
        seed=ep.spec.seed,
        model=model_id,
        messages=ep.messages,
        actions=ep.actions,
        end_reason=ep.end_reason,
        final_files=final_files,
        visible_workspace=_summary(workspace_run),
        visible_original=_summary(original_run),
        hidden=_summary(hidden_run),
    )


def _summary(run: TestRunResult) -> RunSummary:
    return RunSummary(passed=run.passed, failed=run.failed, all_passed=run.all_passed, timed_out=run.timed_out)


def _snapshot(folder: Path, max_bytes: int = 200_000) -> dict[str, str]:
    """Text of every file in the folder (skipping very large ones), keyed by relative path."""
    files = {}
    for path in sorted(folder.rglob("*")):
        if path.is_file() and path.stat().st_size <= max_bytes:
            files[str(path.relative_to(folder))] = path.read_text(encoding="utf-8", errors="replace")
    return files
