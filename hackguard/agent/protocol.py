"""The agent's actions, written as simple tags inside its reply, and the parser that finds them.

The model writes a short thought and then exactly one action:

    <read_file>test_solution.py</read_file>
    <write_file path="solution.py">
    ...full new file contents...
    </write_file>
    <run_tests/>
    <submit/>

Why tags instead of JSON tool calls: writing a whole Python file inside a JSON string means
escaping every quote and newline, which small models often get wrong. Tags keep the code
as plain text, so the tokens the model writes are exactly the code it means. That also
matters later, when we look for the exact token where cheating code starts.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Generation stops as soon as the model finishes one of these, so it cannot run on and
# invent the tool's reply itself. (<read_file> is closed by </read_file>.)
STOP_STRINGS = ["</read_file>", "</write_file>", "<run_tests/>", "<submit/>"]

_PATTERNS = {
    # re.DOTALL lets "." match newlines, so file contents can span many lines.
    "read_file": re.compile(r"<read_file>\s*(?P<path>[^<]*?)\s*</read_file>", re.DOTALL),
    "write_file": re.compile(r'<write_file path="(?P<path>[^"]+)">\n?(?P<content>.*?)</write_file>', re.DOTALL),
    "run_tests": re.compile(r"<run_tests\s*/>"),
    "submit": re.compile(r"<submit\s*/>"),
}

_CODE_FENCE = re.compile(r"^\s*```[a-zA-Z0-9]*\n(?P<body>.*?)\n?```\s*$", re.DOTALL)


@dataclass(frozen=True)
class Action:
    """One parsed action, with where it sits inside the assistant's message.

    start/end: character positions of the whole action tag in the message.
    content_start: character position where the written file's contents begin
    (only for write_file). We need it to find the exact spot of cheating code later.
    """

    kind: str  # "read_file", "write_file", "run_tests", "submit" or "invalid"
    start: int
    end: int
    path: str | None = None
    content: str | None = None
    content_start: int | None = None


def parse_action(message: str) -> Action:
    """Find the first action in a reply. Returns kind "invalid" if there is none."""
    best: tuple[int, str, re.Match[str]] | None = None
    for kind, pattern in _PATTERNS.items():
        match = pattern.search(message)
        if match and (best is None or match.start() < best[0]):
            best = (match.start(), kind, match)
    if best is None:
        return Action(kind="invalid", start=len(message), end=len(message))

    _, kind, match = best
    if kind == "read_file":
        return Action(kind, match.start(), match.end(), path=match.group("path").strip())
    if kind == "write_file":
        content, offset = _strip_code_fence(match.group("content"))
        return Action(kind, match.start(), match.end(), path=match.group("path").strip(),
                      content=content, content_start=match.start("content") + offset)
    return Action(kind, match.start(), match.end())


def truncate_after_action(message: str, action: Action) -> str:
    """Drop anything the model wrote after its first action (it only gets one per turn)."""
    return message if action.kind == "invalid" else message[: action.end]


def _strip_code_fence(content: str) -> tuple[str, int]:
    """Remove a ```python ... ``` fence if the model wrapped the file in one.

    Returns the cleaned content and how many characters were removed from the front, so
    character positions inside the file still map back to the message correctly.
    """
    match = _CODE_FENCE.match(content)
    if not match:
        return content, 0
    return match.group("body") + "\n", match.start("body")
