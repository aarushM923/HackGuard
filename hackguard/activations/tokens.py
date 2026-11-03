"""Turn a trajectory into the model's tokens, and convert between characters and tokens.

The labeler finds the first hacking action as a *character* position inside one message.
The model works on *tokens* (word pieces). To find the token where hacking starts, we:
  1. render the whole conversation with the model's chat template (the exact text the
     model saw), remembering where each message's content sits in that text,
  2. tokenize with "offset mapping", which gives each token's start and end character,
  3. look up which token covers a given character.

Keeping positions as (message number, character inside the message) makes them independent
of any one tokenizer. That is what lets the transfer experiment find "the same spot" in a
second model whose tokenizer splits the text differently.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass
from typing import Any

from hackguard.models import chat_text


@dataclass
class TokenizedTrajectory:
    input_ids: list[int]
    token_ends: list[int]  # end character (in the full text) of each token
    token_starts: list[int]
    message_starts: list[int]  # where each message's content starts in the full text
    message_leads: list[int]  # characters the template trimmed from the front of each message
    message_ends: list[int]
    roles: list[str]

    def assistant_tokens(self) -> list[int]:
        """Indices of tokens that the agent itself wrote (inside assistant messages)."""
        spans = [(s, e) for s, e, r in zip(self.message_starts, self.message_ends, self.roles) if r == "assistant"]
        result = []
        for i, (ts, te) in enumerate(zip(self.token_starts, self.token_ends)):
            if any(s <= ts and te <= e for s, e in spans):
                result.append(i)
        return result

    def char_to_token(self, text_char: int) -> int:
        """Index of the token that contains a character of the full text."""
        # bisect_right finds the first token that ends after this character.
        return min(bisect.bisect_right(self.token_ends, text_char), len(self.token_ends) - 1)

    def message_char_to_text_char(self, message_index: int, char: int) -> int:
        return self.message_starts[message_index] + char - self.message_leads[message_index]

    def message_char_to_token(self, message_index: int, char: int) -> int:
        return self.char_to_token(self.message_char_to_text_char(message_index, char))

    def token_to_message_char(self, token: int) -> tuple[int, int]:
        """(message index, character end inside the message) for a token. -1 if outside messages."""
        end = self.token_ends[token]
        for i, (s, e) in enumerate(zip(self.message_starts, self.message_ends)):
            if s <= end <= e:
                return i, end - s + self.message_leads[i]
        return -1, -1

    def last_token_ending_by(self, message_index: int, char_end: int) -> int:
        """The last token that ends at or before a (message, character) position.

        Used to find "the same spot" in another tokenizer: the token whose text finishes
        closest to (without passing) the character where the original token finished.
        """
        text_end = self.message_char_to_text_char(message_index, char_end)
        return max(bisect.bisect_right(self.token_ends, text_end) - 1, 0)


def tokenize_trajectory(tokenizer: Any, messages: list[dict[str, str]], max_tokens: int | None = None) -> TokenizedTrajectory:
    text = chat_text(tokenizer, messages, add_generation_prompt=False)

    starts, leads, ends = [], [], []
    cursor = 0
    for msg in messages:
        content = msg["content"]
        index = text.find(content, cursor)
        lead = 0
        if index < 0:
            # Some chat templates (Llama's) trim spaces and newlines around each message.
            stripped = content.strip()
            lead = len(content) - len(content.lstrip())
            index = text.find(stripped, cursor)
            if index < 0:
                raise ValueError("could not find a message's text inside the rendered chat")
            content = stripped
        starts.append(index)
        leads.append(lead)
        ends.append(index + len(content))
        cursor = index + len(content)

    # The chat template already contains the special start tokens, so add none here.
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = list(encoded["input_ids"])
    offsets = list(encoded["offset_mapping"])
    if max_tokens is not None:
        ids, offsets = ids[:max_tokens], offsets[:max_tokens]
    # Binary search needs end positions that never go down. Some tokenizers report special
    # tokens as (0, 0), so carry the previous position forward in that case.
    token_starts: list[int] = []
    token_ends: list[int] = []
    last = 0
    for s, e in offsets:
        last = max(int(e), last)
        token_ends.append(last)
        token_starts.append(max(int(s), token_starts[-1] if token_starts else 0))
    return TokenizedTrajectory(
        input_ids=ids,
        token_starts=token_starts,
        token_ends=token_ends,
        message_starts=starts,
        message_leads=leads,
        message_ends=ends,
        roles=[m["role"] for m in messages],
    )
